import logging
import os
import time

import numpy as np
import torch


class FedAVGAggregator(object):

    def __init__(self, model_trainer, args, device):
        self.trainer = model_trainer
        self.args = args
        self.worker_num = args.client_num_in_total
        self.device = device

    def get_global_model_params(self):
        return self.trainer.get_global_model_params()

    def set_global_model_params(self, model_parameters):
        self.trainer.set_global_model_params(model_parameters)

    def aggregate(self, model_params_list, sample_num_list):
        training_num = sum(sample_num_list)
        logging.info("len of self.model_params_list = " + str(len(model_params_list)))
        averaged_params = model_params_list[0]
        for i in range(0, len(model_params_list)):
            local_sample_number = sample_num_list[i]
            local_model_params = model_params_list[i]
            w = local_sample_number / training_num

            logging.info('aggregate model of client %d with w %s' % (i, w))

            for k in averaged_params.keys():
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w

        # update the global model which is cached at the server side
        self.set_global_model_params(averaged_params)

        filename = os.path.join('cache', str(time.time()))
        torch.save(averaged_params, filename)

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def test_on_server(self):
        self.trainer.test()

