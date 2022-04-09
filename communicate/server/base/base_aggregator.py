import logging
from abc import ABC, abstractmethod
import numpy as np


class BaseAggregator(ABC):
    def __init__(self, model_trainer):
        self.trainer = model_trainer

    def get_global_model_params(self):
        return self.trainer.get_global_model_params()

    def set_global_model_params(self, model_parameters):
        self.trainer.set_global_model_params(model_parameters)

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            # np.random.seed(round_idx)
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    @abstractmethod
    def test_on_server(self):
        pass

    @abstractmethod
    def aggregate(self, **kwargs):
        pass
