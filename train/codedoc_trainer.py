import logging
import math
import os

import torch
from tqdm import tqdm
import numpy as np
from transformers import get_linear_schedule_with_warmup, AdamW

from utils import bleu
from utils.model_utils import copy_state_dict


class CodeDocTrainer:
    def __init__(self, args, device, model, tokenizer, train_dl=None, valid_dl=None, test_dl=None):
        self.args = args
        self.device = device

        # self.num_labels = args.num_labels
        self.set_data(train_dl, valid_dl, test_dl)

        self.model = model
        self.tokenizer = tokenizer
        self.global_model_params = copy_state_dict(self.model.state_dict())

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
            log_loss = 0.0
            step = 0
            for batch in tqdm(self.train_dl, desc="training"):

                batch = tuple(t.to(self.device) for t in batch)
                source_ids, source_mask, target_ids, target_mask = batch
                loss, _, _ = model(source_ids=source_ids, source_mask=source_mask, target_ids=target_ids,
                                   target_mask=target_mask)

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                log_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    global_step += 1

                if step % 300 == 0:
                    logging.info("step %s loss = %s" % (step, log_loss / step))

                step += 1
            logging.info("epoch %s loss = %s" % (idx, log_loss / step))
            tr_loss += log_loss

            if args.do_eval:
                self.eval()

        model.cpu()

        return global_step, tr_loss / global_step

    def test(self):
        model = self.model
        tokenizer = self.tokenizer
        model.to(self.device)
        model.eval()
        p = []
        for batch in tqdm(self.test_dl, total=len(self.test_dl)):
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
        dev_bleu = round(bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
        logging.info("  %s = %s " % ("bleu-4", str(dev_bleu)))

        model.cpu()

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
        if dev_bleu > best_bleu:
            logging.info("  Best bleu:%s", dev_bleu)
            logging.info("  " + "*" * 20)
            best_bleu = dev_bleu
            # Save best checkpoint for best bleu
            output_dir = os.path.join(self.args.output_dir, 'checkpoint-best-bleu')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(output_dir, "pytorch_model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)

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
