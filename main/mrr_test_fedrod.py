import argparse
import gc
import logging
import os.path
import pickle
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from more_itertools import chunked
from tqdm import tqdm
from transformers import RobertaConfig, RobertaTokenizer
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

from data.manager.codesearch_data_manager import Example, CodeSearchDataManager
from data.preprocess.base.base_data_loader import BaseDataLoader
from data.preprocess.codesearch_preprocessor import CodeSearchPreprocessor
from main.initialize import add_mrr_test_args, set_seed
from model.roberta_model import RobertaForSequenceClassification, HyperClassifier


def format_str(string):
    for char in ['\r\n', '\r', '\n']:
        string = string.replace(char, ' ')
    return string


def process_data_and_test(test_raw_examples, test_model, preprocessor, args, test_batch_size, p_head_list,
                          label_weight_list,
                          result_weight_list):
    idxs = np.arange(len(test_raw_examples))
    data = np.array(test_raw_examples, dtype=np.object)

    np.random.shuffle(idxs)
    data = data[idxs]
    for i in range(len(result_weight_list)):
        result_weight_list[i] = result_weight_list[i][idxs]
    batched_data = chunked(data, test_batch_size)

    global_ranks = []
    local_ranks = [[] for _ in range(len(p_head_state_list))]

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
        global_preds, local_preds = test(args, data_loader, test_model, p_head_list, label_weight_list)

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

        del batched_logits, global_preds, local_preds, data_loader, examples, features, dataset
        gc.collect()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'fedrod_mrr_test_result.txt'), 'a') as f:
        global_mrr = np.mean(1.0 / np.array(global_ranks))
        print("global mrr: %s" % (global_mrr))
        f.write("TEST TIME:%s\n" % time.asctime(time.localtime(time.time())))
        f.write("global mrr: %s\n\n" % (global_mrr))
        f.write("mrr - global mrr - pre mrr\n\n")
        mrr_list = []
        mrr_global_list = []
        mrr_pre_list = []
        for i, ranks in enumerate(local_ranks):
            mrr_pre = np.mean(1.0 / np.array(ranks))
            mrr = np.sum((1.0 / np.array(ranks)) * result_weight_list[i] / np.sum(result_weight_list[i]))
            mrr_global = np.sum((1.0 / np.array(global_ranks)) * result_weight_list[i] / np.sum(result_weight_list[i]))
            mrr_list.append(mrr)
            mrr_global_list.append(mrr_global)
            mrr_pre_list.append(mrr_pre)
            print('client %s: %s, %s, %s' % (i, mrr, mrr_global, mrr_pre))
            f.write('client %s: %s, %s, %s\n\n' % (i, mrr, mrr_global, mrr_pre))
        print("avg mrr: %s, %s, %s" % (np.mean(mrr_list), np.mean(mrr_global_list), np.mean(mrr_pre_list)))
        f.write("avg mrr: %s, %s, %s" % (np.mean(mrr_list), np.mean(mrr_global_list), np.mean(mrr_pre_list)))
        print("max mrr: %s, %s, %s" % (np.max(mrr_list), np.max(mrr_global_list), np.max(mrr_pre_list)))
        f.write("max mrr: %s, %s, %s" % (np.max(mrr_list), np.max(mrr_global_list), np.max(mrr_pre_list)))
        print("min mrr: %s, %s, %s" % (np.min(mrr_list), np.min(mrr_global_list), np.min(mrr_pre_list)))
        f.write("min mrr: %s, %s, %s" % (np.min(mrr_list), np.min(mrr_global_list), np.min(mrr_pre_list)))


def test(args, data_loader, model, p_head_list, label_weight_list):
    global_preds = None
    local_preds = []
    model.eval()
    for batch in tqdm(data_loader, desc="Testing"):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                      'labels': batch[3]}

            sequence_output = model(**inputs)
            global_logits = model.forward_global(sequence_output)
            local_logits_list = []
            for i, p_head in enumerate(p_head_list):
                local_logits = p_head(label_weight_list[i].to(args.device), sequence_output) + global_logits
                local_logits_list.append(local_logits)

        if global_preds is None:
            global_preds = global_logits.detach().cpu().numpy()
            for local_logits in local_logits_list:
                local_preds.append(local_logits.detach().cpu().numpy())
        else:
            global_preds = np.append(global_preds, global_logits.detach().cpu().numpy(), axis=0)
            for i, local_logits in enumerate(local_logits_list):
                local_preds[i] = np.append(local_preds[i], local_logits.detach().cpu().numpy(), axis=0)

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
    model = model_class.from_pretrained('microsoft/codebert-base', config=config, label_count=10)
    model.load_state_dict(torch.load(os.path.join(args.model_name, 'model.pt')))
    model.to(device)

    p_head_state_list = torch.load(os.path.join(args.model_name, 'p_head.pt'))
    label_weight_list = torch.load(os.path.join(args.model_name, 'label_weight.pt'))
    p_head_list = [HyperClassifier(config.hidden_size, args.label_count) for _ in range(len(p_head_state_list))]
    for p_head, state in zip(p_head_list, p_head_state_list):
        name_state = OrderedDict()
        for key, value in state.items():
            name_state['.'.join(key.split('.')[1:])] = value
        p_head.load_state_dict(name_state)
        p_head.to(device)

    preprocessor = CodeSearchPreprocessor(args, tokenizer)
    manager = CodeSearchDataManager(args, preprocessor)
    test_raw_examples = manager.read_examples_from_jsonl(args.data_file)

    with open(args.label_file, 'rb') as f:
        label_assignment, train_len = pickle.load(f)
    label_assignment = np.array(label_assignment[train_len:])
    sample_idx = []
    not_sample = []
    not_sample_count = 0
    for i in range(args.label_count):
        idx = np.where(label_assignment == i)[0].tolist()
        sample_idx.extend(idx[:20])
        not_sample.extend(idx[20:])
        not_sample_count += max(0, 20 - len(idx))
    sample_idx.extend(not_sample[:not_sample_count])
    with open(os.path.join(args.model_name, 'sample_idx.pt'), 'wb') as f:
        pickle.dump(sample_idx, f)

    test_raw_examples = np.array(test_raw_examples, dtype=np.object)
    test_raw_examples = test_raw_examples[sample_idx]
    label_assignment = label_assignment[sample_idx]
    logging.info('sample test data:%s ,class not sample enough:%s ' % (len(test_raw_examples), not_sample_count))

    result_weight_list = []
    for label_weight in label_weight_list:
        result_weight = []
        label_dict = label_weight.tolist()
        for label in label_assignment:
            result_weight.append(label_dict[label])
        result_weight_list.append(np.array(result_weight))
    process_data_and_test(test_raw_examples, model, preprocessor, args, args.test_batch_size, p_head_list,
                          label_weight_list,
                          result_weight_list)
