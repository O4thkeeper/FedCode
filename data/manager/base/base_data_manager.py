import json
import logging
import os
import pickle
from abc import ABC, abstractmethod

import h5py
import numpy as np
from tqdm import tqdm

from data.preprocess.base.base_data_loader import BaseDataLoader


class BaseDataManager(ABC):
    @abstractmethod
    def __init__(self, args, model_args, num_workers):
        self.model_args = model_args
        self.args = args
        self.train_batch_size = model_args.train_batch_size
        self.eval_batch_size = model_args.eval_batch_size
        self.num_workers = num_workers

        self.num_clients = self.load_num_clients(
            self.args.partition_file_path, self.args.partition_method)

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
    def read_instance_from_h5(self, data_file, index_list, desc=""):
        pass

    def get_all_clients(self):
        return list(range(0, self.num_clients))

    def load_centralized_data(self, cut_off=None):
        state, res = self._load_data_loader_from_cache(-1)
        if state:
            train_examples, train_features, train_dataset, test_examples, test_features, test_dataset = res
        else:
            data_file = h5py.File(self.args.data_file_path, "r", swmr=True)
            partition_file = h5py.File(
                self.args.partition_file_path, "r", swmr=True)
            partition_method = self.args.partition_method
            train_index_list = []
            test_index_list = []
            for client_idx in tqdm(
                    partition_file[partition_method]
                    ["partition_data"].keys(),
                    desc="Loading index from h5 file."):
                train_index_list.extend(
                    partition_file[partition_method]["partition_data"]
                    [client_idx]["train"][()][:cut_off])
                test_index_list.extend(
                    partition_file[partition_method]["partition_data"]
                    [client_idx]["test"][()][:cut_off])
            train_data = self.read_instance_from_h5(data_file, train_index_list)
            test_data = self.read_instance_from_h5(data_file, test_index_list)
            data_file.close()
            partition_file.close()
            train_examples, train_features, train_dataset = self.preprocessor.transform(
                **train_data, index_list=train_index_list)
            test_examples, test_features, test_dataset = self.preprocessor.transform(
                **test_data, index_list=test_index_list, evaluate=True)

            with open(res, "wb") as handle:
                pickle.dump((train_examples, train_features, train_dataset, test_examples, test_features, test_dataset),
                            handle)

        train_dl = BaseDataLoader(train_examples, train_features, train_dataset,
                                  batch_size=self.train_batch_size,
                                  num_workers=0,
                                  pin_memory=True,
                                  drop_last=False)

        test_dl = BaseDataLoader(test_examples, test_features, test_dataset,
                                 batch_size=self.eval_batch_size,
                                 num_workers=0,
                                 pin_memory=True,
                                 drop_last=False)

        return train_dl, test_dl

    def load_federated_data(self, server, test_only=True, test_cut_off=None):
        if server:
            return self._load_federated_data_server(test_only, test_cut_off)
        else:
            return self._load_federated_data_local()

    def _load_federated_data_server(self, test_only=True, test_cut_off=None):
        state, res = self._load_data_loader_from_cache(-1)
        logging.info('load from cache %s , dir: %s' % (str(state), res))
        if state:
            train_examples, train_features, train_dataset, test_examples, test_features, test_dataset = res
            logging.info("load from cache, test data size " + str(len(test_examples)))
        else:
            data_file = h5py.File(self.args.data_file_path, "r", swmr=True)
            partition_file = h5py.File(self.args.partition_file_path, "r", swmr=True)
            partition_method = self.args.partition_method
            train_index_list = []
            test_index_list = []
            for client_idx in tqdm(partition_file[partition_method]["partition_data"].keys(),
                                   desc="Loading index from h5 file."):
                train_index_list.extend(partition_file[partition_method]["partition_data"][client_idx]["train"][()])
                local_test_index_list = partition_file[partition_method]["partition_data"][client_idx]["test"][()]
                test_index_list.extend(local_test_index_list)

            if not test_only:
                train_data = self.read_instance_from_h5(data_file, train_index_list)
            if test_cut_off:
                test_index_list.sort()
            test_index_list = test_index_list[:test_cut_off]
            logging.info("caching test index size " + str(len(test_index_list)) + "test cut off " + str(test_cut_off))

            test_data = self.read_instance_from_h5(data_file, test_index_list)

            data_file.close()
            partition_file.close()

            train_examples, train_features, train_dataset = None, None, None
            if not test_only:
                train_examples, train_features, train_dataset = self.preprocessor.transform(**train_data,
                                                                                            index_list=train_index_list)
            test_examples, test_features, test_dataset = self.preprocessor.transform(**test_data,
                                                                                     index_list=test_index_list)
            logging.info("caching test data size " + str(len(test_examples)))

            with open(res, "wb") as handle:
                pickle.dump((train_examples, train_features, train_dataset, test_examples, test_features, test_dataset),
                            handle)

        if test_only or train_dataset is None:
            train_data_num = 0
            train_data_global = None
        else:
            train_data_global = BaseDataLoader(train_examples, train_features, train_dataset,
                                               batch_size=self.train_batch_size,
                                               num_workers=0,
                                               pin_memory=True,
                                               drop_last=False)
            train_data_num = len(train_examples)
            logging.info("train_dl_global number = " + str(len(train_data_global)))

        test_dataloader = BaseDataLoader(test_examples, test_features, test_dataset,
                                         batch_size=self.eval_batch_size,
                                         num_workers=0,
                                         pin_memory=True,
                                         drop_last=False)

        logging.info("test_dl_global number = " + str(len(test_dataloader)))

        return train_data_global, train_data_num, test_dataloader

    def _load_federated_data_local(self):

        data_file = h5py.File(self.args.data_file_path, "r", swmr=True)
        partition_file = h5py.File(self.args.partition_file_path, "r", swmr=True)
        partition_method = self.args.partition_method

        train_loader_list = []
        train_data_num_list = []
        test_loader_list = []

        for idx in range(self.num_clients):
            state, res = self._load_data_loader_from_cache(idx)
            if state:
                train_examples, train_features, train_dataset, test_examples, test_features, test_dataset = res
            else:
                train_index_list = partition_file[partition_method]["partition_data"][str(idx)]["train"][()]
                test_index_list = partition_file[partition_method]["partition_data"][str(idx)]["test"][()]
                train_data = self.read_instance_from_h5(data_file, train_index_list,
                                                        desc=" train data of client_id=%d [_load_federated_data_local] " % idx)
                test_data = self.read_instance_from_h5(data_file, test_index_list,
                                                       desc=" test data of client_id=%d [_load_federated_data_local] " % idx)

                train_examples, train_features, train_dataset = self.preprocessor.transform(
                    **train_data, index_list=train_index_list)
                test_examples, test_features, test_dataset = self.preprocessor.transform(
                    **test_data, index_list=test_index_list, evaluate=True)

                with open(res, "wb") as handle:
                    pickle.dump(
                        (train_examples, train_features, train_dataset, test_examples, test_features, test_dataset),
                        handle)

            train_loader = BaseDataLoader(train_examples, train_features, train_dataset,
                                          batch_size=self.train_batch_size,
                                          num_workers=0,
                                          pin_memory=True,
                                          drop_last=False)

            test_loader = BaseDataLoader(test_examples, test_features, test_dataset,
                                         batch_size=self.eval_batch_size,
                                         num_workers=0,
                                         pin_memory=True,
                                         drop_last=False)
            train_data_num = len(train_examples)

            train_loader_list.append(train_loader)
            train_data_num_list.append(train_data_num)
            test_loader_list.append(test_loader)

        data_file.close()
        partition_file.close()

        return train_loader_list, train_data_num_list, test_loader_list

    def _load_data_loader_from_cache(self, client_id):
        """
        Different clients has different cache file. client_id = -1 means loading the cached file on server end.
        """
        args = self.args
        model_args = self.model_args
        if not os.path.exists(model_args.cache_dir):
            os.mkdir(model_args.cache_dir)
        cached_features_file = os.path.join(
            model_args.cache_dir, args.model_type + "_" + args.model_name.split("/")[-1] + "_cached_" + str(
                args.max_seq_length) + "_" + model_args.model_class + "_"
                                  + args.dataset + "_" + args.partition_method + "_" + str(client_id)
        )

        if os.path.exists(cached_features_file):
            # and (
            # (not model_args.reprocess_input_data and not model_args.no_cache)
            # or (model_args.use_cached_eval_features and not model_args.no_cache)
            # ):
            logging.info(" Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                train_examples, train_features, train_dataset, test_examples, test_features, test_dataset = pickle.load(
                    handle)
            return True, (train_examples, train_features, train_dataset, test_examples, test_features, test_dataset)
        return False, cached_features_file
