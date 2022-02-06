import logging

import torch

from torch.utils.data import TensorDataset

from data.preprocess.base.base_preprocessor import BasePreprocessor


class CodeDocPreprocessor(BasePreprocessor):
    def __init__(self, args, tokenizer):
        super(CodeDocPreprocessor, self).__init__()
        self.args = args
        self.tokenizer = tokenizer

    def transform(self, examples, stage):
        features = convert_examples_to_features(examples, self.tokenizer, self.args, stage)

        all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        all_source_mask = torch.tensor([f.source_mask for f in features], dtype=torch.long)
        if stage == 'test':
            dataset = TensorDataset(all_source_ids, all_source_mask)
        else:
            all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
            all_target_mask = torch.tensor([f.target_mask for f in features], dtype=torch.long)
            dataset = TensorDataset(all_source_ids, all_source_mask, all_target_ids, all_target_mask)

        return examples, features, dataset


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,

                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask


def convert_examples_to_features(examples, tokenizer, args, stage=None):
    features = []
    for example_index, example in enumerate(examples):
        # source
        source_tokens = tokenizer.tokenize(example.source)[:args.max_seq_length - 2]
        source_tokens = [tokenizer.cls_token] + source_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        source_mask = [1] * (len(source_tokens))
        padding_length = args.max_seq_length - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        source_mask += [0] * padding_length

        # target
        if stage == "test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[:args.max_target_length - 2]
        target_tokens = [tokenizer.cls_token] + target_tokens + [tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] * len(target_ids)
        padding_length = args.max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length
        target_mask += [0] * padding_length

        if example_index < 1:
            if stage == 'train':
                logging.info("*** Example ***")
                logging.info("idx: {}".format(example.idx))

                logging.info("source_tokens: {}".format([x.replace('\u0120', '_') for x in source_tokens]))
                logging.info("source_ids: {}".format(' '.join(map(str, source_ids))))
                logging.info("source_mask: {}".format(' '.join(map(str, source_mask))))

                logging.info("target_tokens: {}".format([x.replace('\u0120', '_') for x in target_tokens]))
                logging.info("target_ids: {}".format(' '.join(map(str, target_ids))))
                logging.info("target_mask: {}".format(' '.join(map(str, target_mask))))

        features.append(
            InputFeatures(
                example_index,
                source_ids,
                target_ids,
                source_mask,
                target_mask,
            )
        )
    return features
