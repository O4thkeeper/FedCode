import json
import pickle
import random

from torch.utils.data import RandomSampler

from data.manager.base.base_data_manager import BaseDataManager
from data.preprocess.base.base_data_loader import BaseDataLoader


class CodeDocDataManager(BaseDataManager):

    def __init__(self, args, preprocessor):
        super().__init__(args)
        self.preprocessor = preprocessor

    def load_centralized_data(self):
        pass

    def _load_federated_data_server(self, data_type, data_file, batch_size):
        state, res = self._load_data_loader_from_cache(-1, data_type)
        if state:
            examples, features, dataset = res
        else:
            data = self._read_examples_from_jsonl(data_file)
            examples, features, dataset = self.preprocessor.transform(data, data_type)
            with open(res, "wb") as handle:
                pickle.dump((examples, features, dataset), handle)
        train_sampler = RandomSampler(dataset)
        data_loader = BaseDataLoader(examples, features, dataset,
                                     sampler=train_sampler,
                                     batch_size=batch_size)
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
                    all_data = self._read_examples_from_jsonl(data_file)
                    data_list = [[] for _ in range(num_clients)]
                    for i, example in enumerate(all_data):
                        data_list[partition_dict[str(i)]].append(example)

                data = data_list[idx]
                if data_type == 'test':
                    data = random.sample(data, min(1000, len(data)))
                examples, features, dataset = self.preprocessor.transform(data, data_type)
                with open(res, "wb") as handle:
                    pickle.dump((examples, features, dataset), handle)

            train_sampler = RandomSampler(dataset)
            data_loader = BaseDataLoader(examples, features, dataset,
                                         sampler=train_sampler,
                                         batch_size=batch_size)
            data_num = len(examples)
            loader_list.append(data_loader)
            data_num_list.append(data_num)

            return loader_list, data_num_list

    def _read_examples_from_jsonl(self, filename):
        examples = []
        with open(filename, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                js = json.loads(line)
                if 'idx' not in js:
                    js['idx'] = idx
                code = ' '.join(js['code_tokens']).replace('\n', ' ')
                code = ' '.join(code.strip().split())
                nl = ' '.join(js['docstring_tokens']).replace('\n', '')
                nl = ' '.join(nl.strip().split())
                examples.append(Example(idx, code, nl))
        return examples


class Example:

    def __init__(self, idx, source, target):
        self.idx = idx
        self.source = source
        self.target = target
