class ClientTrainer(object):

    def __init__(self, train_loader_list, train_data_num_list, test_loader_list, device, args, model_trainer,
                 eval_loader_list=None):
        self.train_loader_list = train_loader_list
        self.train_data_num_list = train_data_num_list
        self.test_loader_list = test_loader_list
        self.eval_loader_list = eval_loader_list

        self.device = device
        self.args = args
        self.trainer = model_trainer

    def update_model(self, weights):
        self.trainer.set_model_params(weights)

    def train(self, index, current_model):
        self.trainer.set_model_params(current_model)
        self.trainer.set_data(self.train_loader_list[index])
        if self.test_loader_list is not None:
            self.trainer.set_data(test_dl=self.test_loader_list[index])
        if self.eval_loader_list is not None:
            self.trainer.set_data(valid_dl=self.eval_loader_list[index])
        self.trainer.train(index)
        model_params = self.trainer.get_model_params()
        return model_params, self.train_data_num_list[index]

    def test(self):
        pass
        # # train data
        # train_metrics = self.trainer.test(self.train_local, self.device, self.args)
        # train_tot_correct, train_num_sample, train_loss = train_metrics['test_correct'], \
        #                                                   train_metrics['test_total'], train_metrics['test_loss']
        #
        # # test data
        # test_metrics = self.trainer.test(self.test_local, self.device, self.args)
        # test_tot_correct, test_num_sample, test_loss = test_metrics['test_correct'], \
        #                                                test_metrics['test_total'], test_metrics['test_loss']
        #
        # return train_tot_correct, train_loss, train_num_sample, test_tot_correct, test_loss, test_num_sample
