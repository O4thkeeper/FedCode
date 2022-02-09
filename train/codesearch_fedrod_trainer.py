import logging
import math
import os

import torch
from tqdm import tqdm
import numpy as np
from transformers import get_linear_schedule_with_warmup, AdamW

from model.roberta_model import BSMLoss
from utils.evaluate_utils import compute_metrics
from utils.model_utils import copy_state_dict


class CodeSearchFedrodTrainer:
    def __init__(self, args, device, model, train_dl=None, valid_dl=None, test_dl=None, h_linear_state_list=None):
        self.args = args
        self.device = device

        self.num_labels = args.num_labels
        self.cls_num_list = args.cls_num_list
        self.set_data(train_dl, valid_dl, test_dl)

        self.model = model
        self.global_model_params = copy_state_dict(self.model.state_dict())
        self.h_linear_state_list = h_linear_state_list

        self.results = {}
        self.best_accuracy = 0.0

        self.freeze_layers = args.freeze_layers.split(",") if args.freeze_layers else []

    def set_data(self, train_dl=None, valid_dl=None, test_dl=None):
        if train_dl is not None:
            self.train_dl = train_dl
        if valid_dl is not None:
            self.valid_dl = valid_dl
        if test_dl is not None:
            self.test_dl = test_dl

    def get_model_params(self):
        return copy_state_dict(self.model.state_dict())

    def set_model_params(self, model_parameters, index):
        model_parameters.update(self.h_linear_state_list[index])
        self.model.load_state_dict(model_parameters)

    def set_global_model_params(self, params):
        self.global_model_params = copy_state_dict(params)

    def get_global_model_params(self):
        return self.global_model_params

    def get_model(self):
        return self.model

    def train(self, index):

        self.model.to(self.device)
        iteration_in_total = len(self.train_dl) // self.args.gradient_accumulation_steps * self.args.epochs
        optimizer, scheduler, ghead_optimizer, phead_optimizer = self.build_optimizer(self.model, iteration_in_total)
        local_loss_fn = torch.nn.CrossEntropyLoss().to(self.device)
        global_loss_fn = BSMLoss(torch.tensor(self.cls_num_list[index]).to(self.device)).to(self.device)

        logging.info("***** Running training *****")

        args = self.args
        global_step = 0
        tr_loss = [0.0, 0.0]
        self.model.zero_grad()

        self.model.train()

        for idx in range(args.epochs):
            log_loss = [0.0, 0.0]
            loss_list = [[], []]
            step = 0
            for batch in tqdm(self.train_dl, desc="training"):
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                          'labels': batch[3]}
                sequence_output = self.model(**inputs)
                logits = self.model.forward_global(sequence_output)
                logits_local = self.model.forward_local_bias(sequence_output.detach())
                labels = batch[3]

                global_loss = global_loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
                global_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                log_loss[0] += global_loss.item()
                optimizer.step()
                scheduler.step()
                ghead_optimizer.step()
                self.model.zero_grad()

                local_loss = local_loss_fn(logits_local.view(-1, self.num_labels), labels.view(-1))
                local_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                log_loss[1] += local_loss.item()
                phead_optimizer.step()
                self.model.zero_grad()

                if step % 100 == 0:
                    loss_list[0].append(global_loss.item())
                    loss_list[1].append(local_loss.item())

                step += 1
            global_step += step
            logging.info(
                "epoch %s train global_loss = %s local_loss = %s" % (idx, log_loss[0] / step, log_loss[1] / step))
            logging.info("epoch %s sample global_loss = %s" % (idx, loss_list[0]))
            logging.info("epoch %s sample local_loss = %s" % (idx, loss_list[1]))

            tr_loss = [a + b for a, b in zip(log_loss, tr_loss)]

            if args.do_eval:
                self.eval(index)
        for name, param in self.model.state_dict():
            if 'h_linear' in name:
                self.h_linear_state_list[index][name] = param.clone().detach().cpu()

        self.model.cpu()

        return global_step, [tl / global_step for tl in tr_loss]

    def test(self):
        self.model.to(self.device)
        # for acc test
        logging.info("***** Running Test *****")
        preds = None
        out_label_ids = None
        for batch in tqdm(self.test_dl, desc="Testing"):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if self.args.model_type in ['bert', 'xlnet'] else None,
                          # XLM don't use segment_ids
                          'labels': batch[3]}

                outputs = self.model(**inputs)
                _, logits = outputs[:2]
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
        preds_label = np.argmax(preds, axis=1)
        result = compute_metrics(preds_label, out_label_ids)
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)
        output_test_file = os.path.join(self.args.output_dir, "test_results.txt")
        with open(output_test_file, "a+") as writer:
            logging.info("***** Test results {} *****")
            for key in sorted(result.keys()):
                logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    def eval(self, index):
        self.model.to(self.device)
        logging.info("***** Running Evaluation *****")
        logging.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = [0.0, 0.0]
        nb_eval_steps = 0
        preds_global = None
        preds_local = None
        out_label_ids = None
        local_loss_fn = torch.nn.CrossEntropyLoss()
        # todo cls_num_list is train data's now
        global_loss_fn = BSMLoss(cls_num_list=self.cls_num_list[index])
        for batch in tqdm(self.valid_dl, desc="Evaluating"):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if self.args.model_type in ['bert', 'xlnet'] else None,
                          'labels': batch[3]}

                sequence_output = self.model(**inputs)
                logits_global = self.model.forward_global(sequence_output)
                logits_local = self.model.forward_local_bias(sequence_output)
                labels = batch[3]
                global_loss = global_loss_fn(logits_global.view(-1, self.num_labels), labels.view(-1))
                local_loss = local_loss_fn(logits_local.view(-1, self.num_labels), labels.view(-1))
                eval_loss[0] += global_loss.item()
                eval_loss[1] += local_loss.item()

            nb_eval_steps += 1
            if preds_global is None:
                preds_global = logits_global.detach().cpu().numpy()
                preds_local = logits_local.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds_global = np.append(preds_global, logits_global.detach().cpu().numpy(), axis=0)
                preds_local = np.append(preds_local, logits_local.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
        eval_loss = [loss / nb_eval_steps for loss in eval_loss]
        preds_label_global = np.argmax(preds_global, axis=1)
        preds_label_local = np.argmax(preds_local, axis=1)
        result_global = compute_metrics(preds_label_global, out_label_ids)
        result_local = compute_metrics(preds_label_local, out_label_ids)

        logging.info(
            "global result: loss: %s; acc: %s; f1: %s" % (eval_loss, result_global['acc'], result_global['f1']))
        logging.info("local result: loss: %s; acc: %s; f1: %s" % (eval_loss, result_local['acc'], result_local['f1']))

    def build_optimizer(self, model, iteration_in_total):
        ghead_optimizer = AdamW(model.classifier.parameters(), lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        phead_optimizer = AdamW(model.h_linear.parameters(), lr=self.args.learning_rate, eps=self.args.adam_epsilon)

        parms = []
        for name, parm in model.state_dict().items():
            if 'h_linear' not in name and 'classifier' not in name:
                parms.append(parm)
        warmup_steps = math.ceil(iteration_in_total * self.args.warmup_ratio)
        logging.info("warmup steps = %d" % warmup_steps)
        self.freeze_model_parameters(model)
        optimizer = AdamW(parms, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=iteration_in_total)
        return optimizer, scheduler, ghead_optimizer, phead_optimizer

    def freeze_model_parameters(self, model):
        modules = list()
        logging.info("freeze layers: %s" % str(self.freeze_layers))
        for layer_idx in self.freeze_layers:
            if layer_idx == "e":
                modules.append(model.distilbert.embeddings)
            else:
                modules.append(model.distilbert.transformer.layer[int(layer_idx)])
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
        logging.info(self.get_parameter_number(model))

    def get_parameter_number(self, net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}
