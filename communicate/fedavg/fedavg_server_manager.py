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
            logging.info('round %d begin' % round)
            client_indexes = self.aggregator.client_sampling(round, self.client_num, self.args.client_num_per_round)
            current_model = self.aggregator.get_global_model_params()

            model_path_list, sample_num_list = self.clients.train(client_indexes, current_model)
            global_model_params = self.aggregator.aggregate(model_path_list, sample_num_list)
            self.aggregator.test_on_server_for_all_clients(round)
