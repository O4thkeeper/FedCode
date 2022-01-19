import argparse
import logging
import os.path

import numpy as np
import torch
from more_itertools import chunked
from tqdm import tqdm
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer

from data.manager.code_search_data_manager import CodeSearchDataManager
from data.preprocess.base.base_data_loader import BaseDataLoader
from data.preprocess.code_search_preprocessor import CodeSearchPreprocessor
from main.initialize import add_mrr_test_args, set_seed

DATA_DIR = '../data/codesearch_data'


def format_str(string):
    for char in ['\r\n', '\r', '\n']:
        string = string.replace(char, ' ')
    return string


def process_data_and_test(test_raw_examples, test_model, preprocessor, args, test_batch_size=1000):
    idxs = np.arange(len(test_raw_examples))
    data = np.array(test_raw_examples, dtype=np.object)

    np.random.shuffle(idxs)
    data = data[idxs]
    batched_data = chunked(data, test_batch_size)

    ranks = []
    num_batch = 0

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
                example = (doc_token, code_token, str(1))
                examples.append(example)

        examples, features, dataset = preprocessor.temp_transform(examples)
        data_loader = BaseDataLoader(examples, features, dataset,
                                     batch_size=args.batch_size,
                                     num_workers=0,
                                     pin_memory=True,
                                     drop_last=False)
        logging.info("***** Running Test %s *****" % batch_idx)
        all_logits = test(args, data_loader, test_model)

        batched_data = chunked(all_logits, test_batch_size)
        for batch_idx, batch_data in enumerate(batched_data):
            num_batch += 1
            correct_score = batch_data[batch_idx][-1]
            scores = np.array([data[-1] for data in batch_data])
            rank = np.sum(scores >= correct_score)
            logging.info("***** %s batch: correct rank %s*****" % (batch_idx, rank))
            ranks.append(rank)

    mean_mrr = np.mean(1.0 / np.array(ranks))
    logging.info("mrr: %s" % (mean_mrr))
    with open(os.path.join(args.output_dir, 'mrr_test_result.txt'), 'a') as f:
        f.write("rank list:%s" % ranks)
        f.write("mrr: %s" % (mean_mrr))


def test(args, data_loader, model):
    preds = None
    for batch in tqdm(data_loader, desc="Testing"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                      'labels': batch[3]}

            outputs = model(**inputs)
            logits = outputs['logits']
        if preds is None:
            preds = logits.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

    return preds.tolist()


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

    config = config_class.from_pretrained(args.model_name, num_labels=2, finetuning_task='codesearch')
    tokenizer = tokenizer_class.from_pretrained(args.model_type)
    model = model_class.from_pretrained(args.model_name, config=config)
    model.to(device)

    preprocessor = CodeSearchPreprocessor(args=args, label_vocab=None, tokenizer=tokenizer)
    manager = CodeSearchDataManager(args, preprocessor, args.data_type, args.data_file,
                                    args.batch_size, None)

    test_raw_examples = manager.load_mrr_test_data()
    process_data_and_test(test_raw_examples, model, preprocessor, args, args.test_batch_size)
