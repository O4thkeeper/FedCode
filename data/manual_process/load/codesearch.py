import argparse
import json
import logging
import os.path

import h5py
from tqdm import tqdm

from data.manual_process.load.base.base_raw_data_loader import TextClassificationRawDataLoader

# todo delete file
class CodeSearchDataLoader(TextClassificationRawDataLoader):
    def __init__(self, data_path, language, data_type):
        super().__init__(data_path)
        self.train_valid_path = 'train_valid'
        self.test_path = 'test'
        self.language = language
        self.data_type = data_type
        self.Y = list()
        self.X['text_a'] = list()
        self.X['text_b'] = list()

    def load_data(self):
        if self.data_type == 'train':
            train_size = self.process_data_file(
                os.path.join(self.data_path, self.train_valid_path, self.language, 'train.txt'))
            self.attributes["train_index_list"] = [i for i in range(train_size)]
            self.attributes["index_list"] = self.attributes["train_index_list"]
            self.attributes["label_vocab"] = {label: i for i, label in enumerate(set(self.Y))}
        elif self.data_type == 'valid':
            valid_size = self.process_data_file(
                os.path.join(self.data_path, self.train_valid_path, self.language, 'valid.txt'))
            self.attributes["valid_index_list"] = [i for i in range(valid_size)]
            self.attributes["index_list"] = self.attributes["valid_index_list"]
            self.attributes["label_vocab"] = {label: i for i, label in enumerate(set(self.Y))}
        elif self.data_type == 'test':
            test_size = self.process_data_file(os.path.join(self.data_path, self.test_path, self.language, 'test.txt'))
            self.attributes["test_index_list"] = [i for i in range(test_size)]
            self.attributes["index_list"] = self.attributes["test_index_list"]
            self.attributes["label_vocab"] = {label: i for i, label in enumerate(set(self.Y))}
        else:
            raise AttributeError('data_type not correct,type should in [train_valid, test]')

    def process_data_file(self, file_path):
        lines = self._read_tsv(file_path)
        if self.data_type == 'test':
            self.X['lines'] = lines
        for (i, line) in enumerate(lines):
            text_a = line[3]
            text_b = line[4]
            if self.data_type == 'test':
                label = '0'
            else:
                label = line[0]
            self.X['text_a'].append(text_a)
            self.X['text_b'].append(text_b)
            self.Y.append(label)

        return len(lines)

    def _read_tsv(self, input_file):
        with open(input_file, "r", encoding='utf-8') as f:
            lines = []
            for line in f.readlines():
                line = line.strip().split('<CODESPLIT>')
                if len(line) != 5:
                    continue
                lines.append(line)
            return lines

    def generate_h5_file(self, file_path):
        f = h5py.File(file_path, "w")
        f["attributes"] = json.dumps(self.attributes)
        for i in tqdm(range(len(self.Y)), desc='writing file '):
            f["X/text_a/" + str(i)] = self.X['text_a'][i]
            f["X/text_b/" + str(i)] = self.X['text_b'][i]
            f["Y/" + str(i)] = self.Y[i]
        f.close()
        logging.info('finish generate h5 file: %s' % file_path)


def add_args(parser):
    parser.add_argument('--data_dir', type=str, default='data/store/codesearch', help='data directory')
    parser.add_argument('--h5_file_path', type=str, default='data/store/codesearch/codesearch_data.h5',
                        help='h5 data file path')
    parser.add_argument('--data_type', type=str, default='train', help='value in train_valid or test')

    args = parser.parse_args()
    return args


# language_list = ['go', 'java', 'javascript', 'php', 'python', 'ruby']
language_list = ['python']
if __name__ == '__main__':
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    args = add_args(argparse.ArgumentParser(description='DataLoader-CodeSearch'))
    for language in language_list:
        loader = CodeSearchDataLoader(args.data_dir, language, args.data_type)
        loader.load_data()
        loader.generate_h5_file(args.h5_file_path + '%s_%s.h5' % (language, args.data_type))
