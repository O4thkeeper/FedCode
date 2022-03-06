import json
import logging
import pickle

from torch.utils.data import RandomSampler
from tqdm import tqdm

from data.manager.base.base_data_manager import BaseDataManager
from data.preprocess.base.base_data_loader import BaseDataLoader


class CodeSearchDataManager(BaseDataManager):

    def __init__(self, args, preprocessor):
        super().__init__(args)
        self.preprocessor = preprocessor

    def load_centralized_data(self):
        pass

    def _load_federated_data_server(self, data_type, data_file, batch_size, max_size=None):
        state, res = self._load_data_loader_from_cache(-1, data_type)
        if state:
            examples, features, dataset = res
        else:
            data = self.read_examples_from_txt(data_file)
            examples, features, dataset = self.preprocessor.transform(data, data_type)
            with open(res, "wb") as handle:
                pickle.dump((examples, features, dataset), handle)
        sampler = RandomSampler(dataset)
        data_loader = BaseDataLoader(examples, features, dataset, sampler=sampler, batch_size=batch_size)
        return data_loader

    def _load_federated_data_local(self, data_type, data_file, batch_size, partition_file):
        with open(partition_file, "rb") as f:
            partition_dict = pickle.load(f)
            num_clients = partition_dict["n_client"]
        loader_list = []
        data_num_list = []
        data_list = []

        for idx in range(num_clients):
            state, res = self._load_data_loader_from_cache(idx, data_type)
            if state:
                examples, features, dataset = res
            else:
                if len(data_list) == 0:
                    all_data = self.read_examples_from_txt(data_file)
                    data_list = [[] for _ in range(num_clients)]
                    for i, example in enumerate(all_data):
                        data_list[partition_dict[str(i)]].append(example)
                logging.info("process client %s load data" % idx)
                data = data_list[idx]
                examples, features, dataset = self.preprocessor.transform(data, data_type)
                with open(res, "wb") as handle:
                    pickle.dump((examples, features, dataset), handle)

            sampler = RandomSampler(dataset)
            data_loader = BaseDataLoader(examples, features, dataset, sampler=sampler, batch_size=batch_size)
            data_num = len(examples)
            loader_list.append(data_loader)
            data_num_list.append(data_num)

        return loader_list, data_num_list

    def read_examples_from_txt(self, filename):
        examples = []
        with open(filename, "r", encoding='utf-8') as f:
            for idx, line in enumerate(f.readlines()):
                line = line.strip().split('<CODESPLIT>')
                if len(line) != 5:
                    continue
                label = line[0]
                text_a = line[3]
                text_b = line[4]
                examples.append(Example(idx, text_a, text_b, label))
        return examples

    def read_examples_from_jsonl(self, filename):
        examples = []
        with open(filename, encoding="utf-8") as f:
            for line in tqdm(f.readlines(), desc='loading mrr test data '):
                data = json.loads(line)
                doc_token = data['docstring_tokens']
                code_token = data['code_tokens']
                examples.append((doc_token, code_token))
        return examples


class Example:

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
