import logging
import math
import os

import torch
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
from transformers import get_linear_schedule_with_warmup

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
        if train_dl is not None:
            self.train_dl = train_dl
        if valid_dl is not None:
            self.valid_dl = valid_dl
        if test_dl is not None:
            self.test_dl = test_dl

    def get_model_params(self):
        return copy_state_dict(self.model.state_dict())

    def set_model_params(self, model_parameters, index=None):
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
        optimizer, scheduler = self.build_optimizer(self.model, iteration_in_total)

        args = self.args
        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        self.model.train()

        for idx in range(args.epochs):
            log_loss = 0.0
            step = 0
            loss_list = []
            bar = tqdm(self.train_dl, total=len(self.train_dl))
            for batch in bar:

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
                bar.set_description("epoch {} loss {}".format(idx, loss.item()))

                log_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
                    global_step += 1

                if step % 300 == 0:
                    loss_list.append(loss.item())

                step += 1
            logging.info("loss list with step 300 = %s" % (loss_list))
            logging.info("epoch %s loss = %s" % (idx, log_loss / step))
            tr_loss += log_loss

        self.model.cpu()

        return global_step, tr_loss / global_step

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
        eval_loss = eval_loss / nb_eval_steps
        preds_label = np.argmax(preds, axis=1)
        result = compute_metrics(preds_label, out_label_ids)
        logging.info("evaluation result { loss: %s; acc: %s; f1: %s }" % (eval_loss, result['acc'], result['f1']))

    def build_optimizer(self, model, iteration_in_total):
        warmup_steps = math.ceil(iteration_in_total * self.args.warmup_ratio)
        # logging.info("warmup steps = %d" % warmup_steps)
        self.freeze_model_parameters(model)
        optimizer = AdamW(model.parameters(), lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=iteration_in_total
        )
        return optimizer, scheduler

    def freeze_model_parameters(self, model):
        logging.info("freeze layers: %s" % str(self.freeze_layers))
        for name, param in model.named_parameters():
            for freeze_layer in self.freeze_layers:
                if freeze_layer in name:
                    param.requires_grad = False
        logging.info(self.get_parameter_number(model))

    def get_parameter_number(self, model):
        total_num = sum(p.numel() for p in model.parameters())
        trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}
