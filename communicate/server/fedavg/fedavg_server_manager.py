import logging
import os.path

import torch


class FedAVGServerManager():
    def __init__(self, aggregator, clients, args):
        self.client_num = args.client_num_in_total
        self.args = args
        self.aggregator = aggregator
        self.round_num = args.comm_round
        self.clients = clients

    def run(self):
        for round in range(self.round_num):
            client_indexes = self.aggregator.client_sampling(round, self.client_num, self.args.client_num_per_round)
            logging.info('round %d begin: sample client %s' % (round, str(client_indexes)))
            current_model = self.aggregator.get_global_model_params()

            if round == 0:
                path = self.args.output_dir
                torch.save(current_model, os.path.join(path, "pre.pt"))


            model_params_list, sample_num_list = self.clients.train(client_indexes, current_model)

            if round == 0:
                path = self.args.output_dir
                for i, model in enumerate(model_params_list):
                    torch.save(model, os.path.join(path, "%s.pt" % i))

            self.aggregator.aggregate(model_params_list, sample_num_list)

            # todo only for test
            if round == 0:
                path = self.args.output_dir
                torch.save(self.aggregator.get_global_model_params(), os.path.join(path, "agg.pt"))

            if self.args.do_eval:
                self.aggregator.eval_global_model()

    def test(self):
        self.aggregator.test_on_server()
