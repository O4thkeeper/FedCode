import argparse
import gzip
import os
import json
import numpy as np
from more_itertools import chunked


def format_str(string):
    for char in ['\r\n', '\r', '\n']:
        string = string.replace(char, ' ')
    return string


def mrr_test_data(language, test_batch_size, data_dir):
    path = os.path.join(data_dir, '{}_test_0.jsonl.gz'.format(language))
    print(path)
    with gzip.open(path, 'r') as pf:
        data = pf.readlines()

    idxs = np.arange(len(data))
    data = np.array(data, dtype=np.object)

    np.random.seed(0)  # set random seed so that random things are reproducible
    np.random.shuffle(idxs)
    data = data[idxs]
    batched_data = chunked(data, test_batch_size)

    print("start processing mrr test data")
    for batch_idx, batch_data in enumerate(batched_data):
        if len(batch_data) < test_batch_size:
            break  # the last batch is smaller than the others, exclude.
        examples = []
        for d_idx, d in enumerate(batch_data):
            line_a = json.loads(str(d, encoding='utf-8'))
            doc_token = ' '.join(line_a['docstring_tokens'])
            for dd in batch_data:
                line_b = json.loads(str(dd, encoding='utf-8'))
                code_token = ' '.join([format_str(token) for token in line_b['code_tokens']])

                example = (str(1), line_a['url'], line_b['url'], doc_token, code_token)
                example = '<CODESPLIT>'.join(example)
                examples.append(example)

        data_path = os.path.join(data_dir, 'test/{}'.format(language))
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        file_path = os.path.join(data_path, 'batch_{}.txt'.format(batch_idx))
        print(file_path)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines('\n'.join(examples))


def acc_test_data(language, data_dir):
    path = os.path.join(data_dir, '{}_test_0.jsonl'.format(language))
    print(path)
    with open(path, 'r') as pf:
        data = pf.readlines()

    idxes = np.arange(len(data))
    data = np.array(data, dtype=np.object)

    np.random.seed(0)  # set random seed so that random things are reproducible
    np.random.shuffle(idxes)
    data = data[idxes]

    print("start processing acc test data")
    examples = []
    idxes = np.arange(len(data))
    shuffled_idxes = np.arange(len(data))
    while np.sum(idxes == shuffled_idxes) > 0:
        np.random.shuffle(shuffled_idxes)
    for i, d in enumerate(data):
        line_a = json.loads(d)
        line_b = json.loads(data[shuffled_idxes[i]])
        doc_token_a = ' '.join(line_a['docstring_tokens'])
        code_token_a = ' '.join([format_str(token) for token in line_a['code_tokens']])
        doc_token_b = ' '.join(line_b['docstring_tokens'])
        code_token_b = ' '.join([format_str(token) for token in line_b['code_tokens']])
        example_true = '<CODESPLIT>'.join((str(1), doc_token_a, code_token_a))
        example_false = '<CODESPLIT>'.join((str(0), doc_token_a, code_token_b))
        examples.append(example_true)
        examples.append(example_false)

    data_path = os.path.join(data_dir, 'test/{}'.format(language))
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    file_path = os.path.join(data_path, 'acc_test_data.txt')
    print(file_path)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(examples))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--language', type=str, default='python')
    parser.add_argument('--type', type=str, default='acc', help='acc or mrr')
    parser.add_argument('--batch_size', type=int, default='1000')
    parser.add_argument('--data_dir', type=str, default='data/store/codesearch')
    args = parser.parse_args()
    if args.type == 'acc':
        acc_test_data(args.language, args.data_dir)
    elif args.type == 'mrr':
        mrr_test_data(args.language, args.batch_size, args.data_dir)
