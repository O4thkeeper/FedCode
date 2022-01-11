import logging


class FedAVGClientManager():
    def __init__(self, args, trainer):
        self.args = args
        self.trainer = trainer

    def train(self, client_indexes, current_model):
        model_params_list = []
        sample_num_list = []
        for index in client_indexes:

            logging.info('client %d ready to train:' % (index))

            model_params, local_sample_num = self.trainer.train(index, current_model)
            model_params_list.append(model_params)
            sample_num_list.append(local_sample_num)
        return model_params_list, sample_num_list
