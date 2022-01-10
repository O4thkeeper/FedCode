import json
import logging
import os
import pickle
from abc import ABC, abstractmethod

import h5py

from data.preprocess.base.base_data_loader import BaseDataLoader


class AbstractDataManager(ABC):
    @abstractmethod
    def __init__(self, args, model_args, data_type, data_path, batch_size, partition_path=None):
        self.data_path = data_path
        self.partition_path = partition_path
        self.data_type = data_type
        self.model_args = model_args
        self.args = args
        self.batch_size = batch_size

    @staticmethod
    def load_attributes(data_path):
        data_file = h5py.File(data_path, "r", swmr=True)
        attributes = json.loads(data_file["attributes"][()])
        data_file.close()
        return attributes

    @staticmethod
    def load_num_clients(partition_file_path, partition_name):
        data_file = h5py.File(partition_file_path, "r", swmr=True)
        num_clients = int(data_file[partition_name]["n_clients"][()])
        data_file.close()
        return num_clients

    @abstractmethod
    def read_instance_from_h5(self, data_file, index_list=None, desc=""):
        pass

    def get_all_clients(self):
        return list(range(0, self.num_clients))

    def load_centralized_data(self):
        state, res = self._load_data_loader_from_cache(-1, self.data_type)
        if state:
            examples, features, dataset = res
        else:
            data_file = h5py.File(self.data_path, "r", swmr=True)
            data = self.read_instance_from_h5(data_file)
            data_file.close()
            examples, features, dataset = self.preprocessor.transform(**data)

            with open(res, "wb") as handle:
                pickle.dump((examples, features, dataset), handle)

        data_loader = BaseDataLoader(examples, features, dataset,
                                     batch_size=self.batch_size,
                                     num_workers=0,
                                     pin_memory=True,
                                     drop_last=False)

        return data_loader

    def load_federated_data(self, server):
        if server:
            return self._load_federated_data_server()
        else:
            self.num_clients = self.load_num_clients(self.partition_path, self.args.partition_method)
            return self._load_federated_data_local()

    def _load_federated_data_server(self):
        state, res = self._load_data_loader_from_cache(-1, self.data_type)
        if state:
            examples, features, dataset = res
        else:
            data_file = h5py.File(self.data_path, "r", swmr=True)
            data = self.read_instance_from_h5(data_file)
            data_file.close()
            examples, features, dataset = self.preprocessor.transform(**data)

            with open(res, "wb") as handle:
                pickle.dump((examples, features, dataset), handle)
        data_loader = BaseDataLoader(examples, features, dataset,
                                     batch_size=self.batch_size,
                                     num_workers=0,
                                     pin_memory=True,
                                     drop_last=False)

        return data_loader

    def _load_federated_data_local(self):

        data_file = h5py.File(self.data_path, "r", swmr=True)
        partition_file = h5py.File(self.partition_path, "r", swmr=True)
        partition_method = self.args.partition_method

        loader_list = []
        data_num_list = []

        for idx in range(self.num_clients):
            state, res = self._load_data_loader_from_cache(idx, self.data_type)
            if state:
                examples, features, dataset = res
            else:
                index_list = partition_file[partition_method]["partition_data"][str(idx)][self.data_type][()]
                data = self.read_instance_from_h5(data_file, index_list,
                                                  desc=" train data of client_id=%d [_load_federated_data_local] " % idx)

                examples, features, dataset = self.preprocessor.transform(**data)

                with open(res, "wb") as handle:
                    pickle.dump((examples, features, dataset), handle)
            loader = BaseDataLoader(examples, features, dataset,
                                    batch_size=self.batch_size,
                                    num_workers=0,
                                    pin_memory=True,
                                    drop_last=False)

            data_num = len(examples)

            loader_list.append(loader)
            data_num_list.append(data_num)

        data_file.close()
        partition_file.close()

        return loader_list, data_num_list

    def _load_data_loader_from_cache(self, client_id, data_type):
        """
        Different clients has different cache file. client_id = -1 means loading the cached file on server end.
        """
        args = self.args
        model_args = self.model_args
        if not os.path.exists(model_args.cache_dir):
            os.mkdir(model_args.cache_dir)
        cached_features_file = os.path.join(
            model_args.cache_dir, args.model_type + "_" + args.model_name.split("/")[-1] + "_cached_" + str(
                args.max_seq_length) + "_" + model_args.model_class + "_" + args.dataset + "_" + args.partition_method + "_" + str(
                client_id) + "_" + data_type)

        if os.path.exists(cached_features_file):
            # and (
            # (not model_args.reprocess_input_data and not model_args.no_cache)
            # or (model_args.use_cached_eval_features and not model_args.no_cache)
            # ):
            logging.info(" Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                examples, features, dataset = pickle.load(handle)
            return True, (examples, features, dataset)
        return False, cached_features_file
