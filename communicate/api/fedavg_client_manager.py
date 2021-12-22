import logging


class FedAVGClientManager():
    def __init__(self, args, trainer):
        self.args = args
        self.trainer = trainer

    def train(self, client_indexes, current_model):
        model_path_list = []
        sample_num_list = []
        for index in client_indexes:

            # logging.info('client %d' % (index))
            # for idx, param in enumerate(current_model):
            #     logging.info("%s:%s" % (param, current_model[param][:20]))
            #     break

            path, local_sample_num = self.trainer.train(index, current_model)
            model_path_list.append(path)
            sample_num_list.append(local_sample_num)
        return model_path_list, sample_num_list
