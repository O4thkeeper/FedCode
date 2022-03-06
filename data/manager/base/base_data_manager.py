import logging
import os
import pickle
from abc import ABC, abstractmethod


class BaseDataManager(ABC):

    def __init__(self, args):
        self.args = args

    @abstractmethod
    def load_centralized_data(self):
        pass

    def load_federated_data(self, server, data_type, data_file, batch_size, partition_file=None, max_size=None):
        if server:
            return self._load_federated_data_server(data_type, data_file, batch_size, max_size)
        else:
            return self._load_federated_data_local(data_type, data_file, batch_size, partition_file)

    @abstractmethod
    def _load_federated_data_server(self, data_type, data_file, batch_size, max_size=None):
        pass

    @abstractmethod
    def _load_federated_data_local(self, data_type, data_file, batch_size, partition_file):
        pass

    def _load_data_loader_from_cache(self, client_id, data_type):
        args = self.args
        if not os.path.exists(args.cache_dir):
            os.makedirs(args.cache_dir)
        cached_features_file = os.path.join(args.cache_dir,
                                            args.model_type + "_" + args.model_name.split("/")[-1] + "_cached_"
                                            + str(args.max_seq_length) + "_" + args.dataset + "_"
                                            + args.partition_method + "_" + str(client_id) + "_" + data_type)

        if os.path.exists(cached_features_file):
            logging.info(" Loading from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                examples, features, dataset = pickle.load(handle)
            return True, (examples, features, dataset)
        return False, cached_features_file
