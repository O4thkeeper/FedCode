import logging
import os
import time
import torch

from communicate.server.base.base_aggregator import BaseAggregator


class FedAVGAggregator(BaseAggregator):

    def __init__(self, model_trainer, args, device):
        super().__init__(model_trainer)
        self.args = args
        self.worker_num = args.client_num_in_total
        self.device = device

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
        self.trainer.set_model_params(averaged_params)


    def test_on_server(self):
        self.trainer.test()

    def eval_global_model(self):
        self.trainer.eval()
