import pandas as pd
import torch

from torch.utils.data import TensorDataset

from data.preprocess.base.base_example import TextClassificationInputExample
from data.preprocess.base.base_preprocessor import BasePreprocessor
from data.preprocess.utils.code_search_utils import convert_examples_to_features


class CodeSearchPreprocessor(BasePreprocessor):
    def __init__(self, **kwargs):
        super(CodeSearchPreprocessor, self).__init__(**kwargs)

    def transform(self, X, y):
        examples = self.transform_examples(X, y)
        features = self.transform_features(examples)

        # all_guid = torch.tensor([f.guid for f in features], dtype=torch.long)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        return examples, features, dataset

    def transform_examples(self, X, y):
        data = [(X['text_a'][i], X['text_b'][i], y[i], i) for i in range(len(y))]
        df = pd.DataFrame(data)
        examples = []
        for i, (text_a, text_b, label, guid) in enumerate(
                zip(df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2], df.iloc[:, 3])):
            examples.append(TextClassificationInputExample(guid, text_a, text_b, label))

        return examples

    def transform_features(self, examples):
        label_list = self.get_labels()
        output_mode = "classification"
        features = convert_examples_to_features(examples, label_list, self.args.max_seq_length, self.tokenizer,
                                                output_mode, cls_token_at_end=bool(self.args.model_type in ['xlnet']),
                                                # xlnet has a cls token at the end
                                                cls_token=self.tokenizer.cls_token, sep_token=self.tokenizer.sep_token,
                                                cls_token_segment_id=2 if self.args.model_type in ['xlnet'] else 1,
                                                pad_on_left=bool(self.args.model_type in ['xlnet']),
                                                # pad on the left for xlnet
                                                pad_token_segment_id=4 if self.args.model_type in ['xlnet'] else 0)
        # if args.local_rank in [-1, 0]:
        # logger.info("Saving features into cached file %s", cached_features_file)
        # torch.save(features, cached_features_file)
        return features

    def get_labels(self):
        """See base class."""
        return ["0", "1"]
