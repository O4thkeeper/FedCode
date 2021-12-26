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
        return self.trainer.get_model_params()

    def set_global_model_params(self, model_parameters):
        self.trainer.set_model_params(model_parameters)


    def aggregate(self, model_path_list, sample_num_list):
        start_time = time.time()
        training_num = sum(sample_num_list)

        logging.info("len of self.model_path_list = " + str(len(model_path_list)))

        averaged_params = torch.load(model_path_list[0])
        # for idx, param in enumerate(averaged_params):
        #     logging.info("%s:%s" % (param, averaged_params[param][:20]))
        #     break
        for i in range(0, len(model_path_list)):
            local_sample_number = sample_num_list[i]
            local_model_params = torch.load(model_path_list[i])
            os.remove(model_path_list[i])
            w = local_sample_number / training_num

            # logging.info('client %d' % (i))
            # for idx, param in enumerate(local_model_params):
            #     logging.info("%s:%s" % (param, local_model_params[param][:20]))
            #     break

            for k in averaged_params.keys():
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w

        # update the global model which is cached at the server side
        self.set_global_model_params(averaged_params)

        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))

        filename = os.path.join('cache', str(time.time()))
        torch.save(averaged_params, filename)

        return averaged_params

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def test_on_server_for_all_clients(self, round_idx):
        if not self.trainer.test_on_the_server(self.device):
            logging.info("round %d not tested all" % round_idx)
        return
