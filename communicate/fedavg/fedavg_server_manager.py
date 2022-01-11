import logging


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

            model_params_list, sample_num_list = self.clients.train(client_indexes, current_model)
            self.aggregator.aggregate(model_params_list, sample_num_list)
            # self.aggregator.test_on_server_for_all_clients(round)
