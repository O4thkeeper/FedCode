import argparse
import logging
import math
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, AdamW, \
    get_linear_schedule_with_warmup

from data.manager.codesearch_data_manager import CodeSearchDataManager
from data.preprocess.codesearch_preprocessor import CodeSearchPreprocessor
from main.initialize import set_seed, add_code_search_args
from utils.evaluate_utils import compute_metrics


def train(model, train_loader, eval_loader, args, device):
    iteration_in_total = len(train_loader) // args.gradient_accumulation_steps * args.epochs
    optimizer, scheduler = build_optimizer(model, iteration_in_total, args)

    model.zero_grad()
    model.train()

    check = [i for i in range(len(train_loader) - 1, -1, -int(len(train_loader) / 8))][:8]
    logging.info("eval point:%s" % check)
    eval_result_list = []

    for idx in range(args.epochs):
        logging.info('epoch %s begin' % idx)
        log_loss = 0.0
        step = 0
        bar = tqdm(train_loader, total=len(train_loader))
        for batch in bar:

            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                      'labels': batch[3]}
            ouputs = model(**inputs)
            loss = ouputs[0]

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            bar.set_description("epoch {} loss {}".format(idx, loss.item()))

            log_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                model.zero_grad()

            if step in check:
                eval_result = eval(model, eval_loader, args, device)
                eval_result_list.append(eval_result)
            step += 1
        logging.info("epoch %s loss = %s" % (idx, log_loss / step))
    logging.info("eval result: %s" % eval_result_list)


def eval(model, eval_loader, args, device):
    model.to(device)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_loader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                      'labels': batch[3]}

            outputs = model(**inputs)
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
    model.train()
    return eval_loss, result['acc'], result['f1']


def build_optimizer(model, iteration_in_total, args):
    warmup_steps = math.ceil(iteration_in_total * args.warmup_ratio)
    logging.info("warmup steps = %d" % warmup_steps)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=iteration_in_total)
    return optimizer, scheduler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_code_search_args(parser)
    args = parser.parse_args()

    # customize the log format
    logging.basicConfig(
        level=logging.INFO,
        format='%(process)s %(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s',
        datefmt='%Y-%m-%d,%H:%M:%S')
    logging.info(args)

    set_seed(args.manual_seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config_class, model_class, tokenizer_class = RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer

    if args.do_train:

        config = config_class.from_pretrained(args.model_name, num_labels=2, finetuning_task='codesearch')
        tokenizer = tokenizer_class.from_pretrained(args.model_type)
        model = model_class.from_pretrained(args.model_name, config=config)
        model.to(device)

        # data
        preprocessor = CodeSearchPreprocessor(args, tokenizer)
        manager = CodeSearchDataManager(args, preprocessor)

        train_loader = manager.load_federated_data(True, 'train', args.train_data_file, args.train_batch_size)
        eval_loader = None
        if args.do_eval:
            eval_loader = manager.load_federated_data(True, 'eval', args.eval_data_file, args.eval_batch_size)

        train(model, train_loader, eval_loader, args, device)

        save_dir = os.path.join(args.cache_dir, "model", args.fl_algorithm)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
