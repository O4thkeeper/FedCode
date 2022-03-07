import argparse
import logging
import os.path
import time

import numpy as np
import torch
from more_itertools import chunked
from tqdm import tqdm
from transformers import RobertaConfig, RobertaTokenizer

from data.manager.codesearch_data_manager import Example, CodeSearchDataManager
from data.preprocess.base.base_data_loader import BaseDataLoader
from data.preprocess.codesearch_preprocessor import CodeSearchPreprocessor
from main.initialize import add_mrr_test_args, set_seed
from model.roberta_model import RobertaForSequenceClassification, HyperClassifier


def format_str(string):
    for char in ['\r\n', '\r', '\n']:
        string = string.replace(char, ' ')
    return string


def process_data_and_test(test_raw_examples, test_model, preprocessor, args, test_batch_size, h_linear_state_list,
                          label_weight_list):
    idxs = np.arange(len(test_raw_examples))
    data = np.array(test_raw_examples, dtype=np.object)

    np.random.shuffle(idxs)
    data = data[idxs]
    batched_data = chunked(data, test_batch_size)

    global_ranks = []
    local_ranks = [[] for _ in range(len(h_linear_state_list))]

    for batch_idx, batch_data in enumerate(batched_data):
        if len(batch_data) < test_batch_size:
            break
        examples = []
        for d_idx, d in enumerate(batch_data):
            doc_token, _ = d
            doc_token = ' '.join(doc_token)
            for dd in batch_data:
                _, code_token = dd
                code_token = ' '.join([format_str(token) for token in code_token])
                example = Example(d_idx, doc_token, code_token, str(1))
                examples.append(example)

        examples, features, dataset = preprocessor.transform(examples)
        data_loader = BaseDataLoader(examples, features, dataset,
                                     batch_size=args.batch_size,
                                     num_workers=0,
                                     pin_memory=True,
                                     drop_last=False)
        logging.info("***** Running Test %s *****" % batch_idx)
        global_preds, local_preds = test(args, data_loader, test_model, h_linear_state_list, label_weight_list)

        batched_logits = chunked(global_preds, test_batch_size)
        for batch_idx, batch_data in enumerate(batched_logits):
            correct_score = batch_data[batch_idx][-1]
            scores = np.array([data[-1] for data in batch_data])
            rank = np.sum(scores >= correct_score)
            global_ranks.append(rank)
        for i, preds in enumerate(local_preds):
            batched_logits = chunked(preds, test_batch_size)
            for batch_idx, batch_data in enumerate(batched_logits):
                correct_score = batch_data[batch_idx][-1]
                scores = np.array([data[-1] for data in batch_data])
                rank = np.sum(scores >= correct_score)
                local_ranks[i].append(rank)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'fedrod_mrr_test_result.txt'), 'a') as f:
        global_mrr = np.mean(1.0 / np.array(global_ranks))
        logging.info("global mrr: %s" % (global_mrr))
        f.write("TEST TIME:%s\n" % time.asctime(time.localtime(time.time())))
        f.write("global mrr: %s\n\n" % (global_mrr))
        for i, ranks in enumerate(local_ranks):
            mrr = np.mean(1.0 / np.array(ranks))
            logging.info("client %s mrr: %s" % (i, mrr))
            f.write("client %s mrr: %s\n\n" % (i, mrr))


def test(args, data_loader, model, h_linear_state_list, label_weight_list):
    global_preds = None
    local_preds = []
    mat_list = []
    for i, h_linear_state in enumerate(h_linear_state_list):
        state = model.state_dict()
        state.update(h_linear_state)
        model.load_state_dict(state)
        mat_list.append(model.h_linear(label_weight_list[i].to(args.device)))
    for batch in tqdm(data_loader, desc="Testing"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                      'labels': batch[3]}

            sequence_output = model(**inputs)
            global_logits = model.forward_global(sequence_output)
            local_logits_list = []
            for i, mat in enumerate(mat_list):
                local_logits = torch.matmul(sequence_output.detach()[:, 0, :], mat) + global_logits.detach()
                local_logits_list.append(local_logits)

        if not global_preds:
            global_preds = global_logits.detach().cpu().numpy()
            for local_logits in local_logits_list:
                local_preds.append(local_logits.detach().cpu().numpy())
        else:
            global_preds = np.append(global_preds, global_logits.detach().cpu().numpy(), axis=0)
            for i, local_logits in enumerate(local_logits_list):
                local_preds[i] = np.append(local_preds[i], local_logits_list[i].detach().cpu().numpy(), axis=0)

    return global_preds.tolist(), [preds.tolist() for preds in local_preds]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_mrr_test_args(parser)
    args = parser.parse_args()

    # customize the log format
    logging.basicConfig(
        level=logging.INFO,
        format='%(process)s %(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s',
        datefmt='%Y-%m-%d,%H:%M:%S')
    logging.info(args)

    set_seed(args.manual_seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device

    config_class, model_class, tokenizer_class = RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer

    config = config_class.from_pretrained('microsoft/codebert-base', num_labels=2, finetuning_task='codesearch')
    tokenizer = tokenizer_class.from_pretrained(args.model_type)
    model = model_class.from_pretrained('microsoft/codebert-base', config=config)
    model.load_state_dict(torch.load(os.path.join(args.model_name, 'model.pt')))
    model.to(device)

    h_linear_state_list = torch.load(os.path.join(args.model_name, 'h_linear.pt'))
    label_weight_list = torch.load(os.path.join(args.model_name, 'label_weight.pt'))

    preprocessor = CodeSearchPreprocessor(args, tokenizer)
    manager = CodeSearchDataManager(args, preprocessor)

    test_raw_examples = manager.read_examples_from_jsonl(args.data_file)
    process_data_and_test(test_raw_examples, model, preprocessor, args, args.test_batch_size, h_linear_state_list,
                          label_weight_list)
