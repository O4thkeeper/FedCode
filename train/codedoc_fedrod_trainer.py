import logging
import math
import os

import torch
from tqdm import tqdm
import numpy as np
from transformers import get_linear_schedule_with_warmup, AdamW

from utils import bleu
from utils.model_utils import copy_state_dict


class CodeDocFedRodTrainer:
    def __init__(self, args, device, model, tokenizer, train_dl=None, valid_dl=None, test_dl=None,
                 p_head_state_list=None, vocab_weight_list=None):
        self.args = args
        self.device = device

        self.set_data(train_dl, valid_dl, test_dl)

        self.model = model
        self.tokenizer = tokenizer
        self.global_model_params = copy_state_dict(self.model.state_dict())
        self.p_head_state_list = p_head_state_list
        self.vocab_weight_list = vocab_weight_list

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
        if index is not None and self.p_head_state_list is not None:
            model_parameters.update(self.p_head_state_list[index])
        self.model.load_state_dict(model_parameters)

    def set_global_model_params(self, params):
        self.global_model_params = copy_state_dict(params)

    def get_global_model_params(self):
        return self.global_model_params

    def get_model(self):
        return self.model

    def train(self, index):

        model = self.model
        model.to(self.device)
        iteration_in_total = len(self.train_dl) // self.args.gradient_accumulation_steps * self.args.epochs
        optimizer, scheduler = self.build_optimizer(self.model, iteration_in_total)

        logging.info("***** Running training *****")

        args = self.args
        global_step = 0
        tr_loss = 0.0
        model.zero_grad()
        model.train()

        for idx in range(args.epochs):
            log_loss = [0.0, 0.0]
            step = 0
            loss_list = [[], []]
            bar = tqdm(self.train_dl, total=len(self.train_dl))
            for batch in bar:

                batch = tuple(t.to(self.device) for t in batch)
                source_ids, source_mask, target_ids, target_mask = batch
                loss, _, _, local_loss = model(source_ids=source_ids, source_mask=source_mask, target_ids=target_ids,
                                               target_mask=target_mask, train_p_head=True,
                                               vocab_weight=self.vocab_weight_list[index].to(self.device))

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                    local_loss = local_loss / args.gradient_accumulation_steps

                loss.backward()
                local_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                bar.set_description("epoch {} global_loss {} local_loss {}".format(idx, loss.item(), local_loss.item()))

                log_loss[0] += loss.item()
                log_loss[1] += local_loss.item()

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    global_step += 1

                if step % 300 == 0:
                    loss_list[0].append(loss.item())
                    loss_list[1].append(local_loss.item())

                step += 1
            logging.info("loss list with step 300 = %s" % loss_list)
            logging.info("epoch %s global_loss = %s local_loss = %s" % (idx, log_loss[0] / step, log_loss[1] / step))
            tr_loss += log_loss[0]

        model.cpu()

        return global_step, tr_loss / global_step

    def test(self, index=None):
        model = self.model
        tokenizer = self.tokenizer
        model.to(self.device)
        model.eval()
        p = []
        for batch in tqdm(self.test_dl, total=len(self.test_dl)):
            batch = tuple(t.to(self.device) for t in batch)
            source_ids, source_mask = batch
            with torch.no_grad():
                encoder_output = model(source_ids=source_ids, source_mask=source_mask, return_encode=True)
                if not index:
                    preds = model.predict(encoder_output, source_ids, source_mask)
                else:
                    preds = model.predict(encoder_output, source_ids, source_mask,
                                          self.vocab_weight_list[index].to(self.device))
                for pred in preds:
                    t = pred[0].cpu().numpy()
                    t = list(t)
                    if 0 in t:
                        t = t[:t.index(0)]
                    text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                    p.append(text)
        model.train()
        predictions = []
        with open(os.path.join(self.args.output_dir, "test.output"), 'w') as f, open(
                os.path.join(self.args.output_dir, "test.gold"), 'w') as f1:
            for ref, gold in zip(p, self.test_dl.examples):
                predictions.append(str(gold.idx) + '\t' + ref)
                f.write(str(gold.idx) + '\t' + ref + '\n')
                f1.write(str(gold.idx) + '\t' + gold.target + '\n')

        (goldMap, predictionMap) = bleu.computeMaps(predictions,
                                                    os.path.join(self.args.output_dir, "test.gold"))
        dev_bleu = bleu.bleuFromMaps(goldMap, predictionMap)[0]
        if not index:
            logging.info("global %s = %s " % ("bleu-4", str(dev_bleu)))
        else:
            logging.info("client %s %s = %s " % (index, "bleu-4", str(dev_bleu)))

        model.cpu()
        return dev_bleu

    def eval(self):
        model = self.model
        tokenizer = self.tokenizer
        model.to(self.device)
        model.eval()
        logging.info("***** Running Evaluation *****")
        eval_loss, tokens_num = 0, 0
        best_bleu, best_loss = 0, 1e6
        for batch in tqdm(self.valid_dl, desc="Evaluating loss"):
            batch = tuple(t.to(self.device) for t in batch)
            source_ids, source_mask, target_ids, target_mask = batch
            with torch.no_grad():
                _, loss, num = model(source_ids=source_ids, source_mask=source_mask,
                                     target_ids=target_ids, target_mask=target_mask)
            eval_loss += loss.sum().item()
            tokens_num += num.sum().item()

        eval_loss = eval_loss / tokens_num
        logging.info("eval_ppl = %s" % np.exp(eval_loss))

        p = []
        for batch in tqdm(self.test_dl, desc="Evaluating bleu"):
            batch = tuple(t.to(self.device) for t in batch)
            source_ids, source_mask = batch
            with torch.no_grad():
                preds = model(source_ids=source_ids, source_mask=source_mask)
                for pred in preds:
                    t = pred[0].cpu().numpy()
                    t = list(t)
                    if 0 in t:
                        t = t[:t.index(0)]
                    text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                    p.append(text)
        predictions = []
        with open(os.path.join(self.args.output_dir, "dev.output"), 'w') as f, open(
                os.path.join(self.args.output_dir, "dev.gold"), 'w') as f1:
            for ref, gold in zip(p, self.test_dl.examples):
                predictions.append(str(gold.idx) + '\t' + ref)
                f.write(str(gold.idx) + '\t' + ref + '\n')
                f1.write(str(gold.idx) + '\t' + gold.target + '\n')

        (goldMap, predictionMap) = bleu.computeMaps(predictions, os.path.join(self.args.output_dir, "dev.gold"))
        dev_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
        logging.info("  %s = %s " % ("bleu-4", str(dev_bleu)))
        logging.info("  " + "*" * 20)

        model.cpu()

    def build_optimizer(self, model, iteration_in_total):
        args = self.args
        warmup_steps = math.ceil(iteration_in_total * self.args.warmup_ratio)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmup_steps,
                                                    num_training_steps=iteration_in_total)

        return optimizer, scheduler
