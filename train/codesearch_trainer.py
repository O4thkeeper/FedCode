import logging
import math
import os
import time
from collections import OrderedDict

import torch
from tqdm import tqdm, trange
import numpy as np
from transformers import get_linear_schedule_with_warmup, AdamW

from utils.evaluate_utils import compute_metrics
from utils.model_utils import copy_state_dict


class CodeSearchTrainer:
    def __init__(self, args, device, model, train_dl=None, valid_dl=None, test_dl=None):
        self.args = args
        self.device = device

        # self.num_labels = args.num_labels
        self.set_data(train_dl, valid_dl, test_dl)

        self.model = model
        self.global_model_params = copy_state_dict(self.model.state_dict())

        self.results = {}
        self.best_accuracy = 0.0

        self.freeze_layers = args.freeze_layers.split(",") if args.freeze_layers else []

    def set_data(self, train_dl=None, valid_dl=None, test_dl=None):
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.test_dl = test_dl

    def get_model_params(self):
        return copy_state_dict(self.model.state_dict())

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def set_global_model_params(self, params):
        self.global_model_params = copy_state_dict(params)

    def get_global_model_params(self):
        return self.global_model_params

    def get_model(self):
        return self.model

    def train(self):

        self.model.to(self.device)
        iteration_in_total = len(self.train_dl) // self.args.gradient_accumulation_steps * self.args.epochs
        optimizer, scheduler = self.build_optimizer(self.model, iteration_in_total)

        logging.info("***** Running training *****")

        # global_step = args.start_step
        args = self.args
        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        self.model.train()

        for idx in range(args.epochs):
            log_loss = 0.0
            step = 0
            for batch in tqdm(self.train_dl, desc="training"):

                batch = tuple(t.to(self.device) for t in batch)
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                          'labels': batch[3]}
                ouputs = self.model(**inputs)
                loss = ouputs[0]

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                log_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                step += 1
            logging.info("epoch %s loss = %s" % (idx, log_loss / step))
            tr_loss += log_loss

        self.model.cpu()

        return global_step, tr_loss / global_step

    # def eval(self):
    #     # Loop to handle MNLI double evaluation (matched, mis-matched)
    #     eval_task_names = (args.task_name,)
    #     eval_outputs_dirs = (args.output_dir,)
    #
    #     results = {}
    #     for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
    #         if (mode == 'dev'):
    #             eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, ttype='dev')
    #         elif (mode == 'test'):
    #             eval_dataset, instances = load_and_cache_examples(args, eval_task, tokenizer, ttype='test')
    #
    #         if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
    #             os.makedirs(eval_output_dir)
    #
    #         args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    #         # Note that DistributedSampler samples randomly
    #         eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(
    #             eval_dataset)
    #         eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    #
    #         # Eval!
    #         logging.info("***** Running evaluation {} *****".format(prefix))
    #         logging.info("  Num examples = %d", len(eval_dataset))
    #         logging.info("  Batch size = %d", args.eval_batch_size)
    #         eval_loss = 0.0
    #         nb_eval_steps = 0
    #         preds = None
    #         out_label_ids = None
    #         for batch in tqdm(eval_dataloader, desc="Evaluating"):
    #             self.model.eval()
    #             batch = tuple(t.to(args.device) for t in batch)
    #
    #             with torch.no_grad():
    #                 inputs = {'input_ids': batch[0],
    #                           'attention_mask': batch[1],
    #                           'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
    #                           # XLM don't use segment_ids
    #                           'labels': batch[3]}
    #
    #                 outputs = model(**inputs)
    #                 tmp_eval_loss, logits = outputs[:2]
    #
    #                 eval_loss += tmp_eval_loss.mean().item()
    #             nb_eval_steps += 1
    #             if preds is None:
    #                 preds = logits.detach().cpu().numpy()
    #                 out_label_ids = inputs['labels'].detach().cpu().numpy()
    #             else:
    #
    #                 preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
    #
    #                 out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
    #         # eval_accuracy = accuracy(preds,out_label_ids)
    #         eval_loss = eval_loss / nb_eval_steps
    #         if args.output_mode == "classification":
    #             preds_label = np.argmax(preds, axis=1)
    #         result = compute_metrics(eval_task, preds_label, out_label_ids)
    #         results.update(result)
    #         if (mode == 'dev'):
    #             output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    #             with open(output_eval_file, "a+") as writer:
    #                 logging.info("***** Eval results {} *****".format(prefix))
    #                 writer.write('evaluate %s\n' % checkpoint)
    #                 for key in sorted(result.keys()):
    #                     logging.info("  %s = %s", key, str(result[key]))
    #                     writer.write("%s = %s\n" % (key, str(result[key])))
    #         elif (mode == 'test'):
    #             output_test_file = args.test_result_dir
    #             output_dir = os.path.dirname(output_test_file)
    #             if not os.path.exists(output_dir):
    #                 os.makedirs(output_dir)
    #             with open(output_test_file, "w") as writer:
    #                 logging.info("***** Output test results *****")
    #                 all_logits = preds.tolist()
    #                 for i, logit in tqdm(enumerate(all_logits), desc='Testing'):
    #                     instance_rep = '<CODESPLIT>'.join(
    #                         [item.encode('ascii', 'ignore').decode('ascii') for item in instances[i]])
    #
    #                     writer.write(instance_rep + '<CODESPLIT>' + '<CODESPLIT>'.join([str(l) for l in logit]) + '\n')
    #                 for key in sorted(result.keys()):
    #                     print("%s = %s" % (key, str(result[key])))
    #
    #     return results

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

    def eval(self):
        self.model.to(self.device)
        logging.info("***** Running Evaluation *****")
        logging.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(self.valid_dl, desc="Evaluating"):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if self.args.model_type in ['bert', 'xlnet'] else None,
                          # XLM don't use segment_ids
                          'labels': batch[3]}

                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
        # eval_accuracy = accuracy(preds,out_label_ids)
        eval_loss = eval_loss / nb_eval_steps
        preds_label = np.argmax(preds, axis=1)
        result = compute_metrics(preds_label, out_label_ids)
        logging.info("evaluation result { loss: %s; acc: %s; f1: %s" % (eval_loss, result['acc'], result['f1']))

    def build_optimizer(self, model, iteration_in_total):
        warmup_steps = math.ceil(iteration_in_total * self.args.warmup_ratio)
        # self.args.warmup_steps = warmup_steps if self.args.warmup_steps == 0 else self.args.warmup_steps
        logging.info("warmup steps = %d" % warmup_steps)
        self.freeze_model_parameters(model)
        optimizer = AdamW(model.parameters(), lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        # logging.info('warmup steps:%d' % warmup_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=iteration_in_total
        )
        return optimizer, scheduler

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
