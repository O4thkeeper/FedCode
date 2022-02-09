import argparse
import logging

import torch
from transformers import RobertaConfig, RobertaTokenizer

from data.manager.base.abstract_data_manager import AbstractDataManager
from data.manager.code_search_data_manager import CodeSearchDataManager
from data.preprocess.code_search_preprocessor import CodeSearchPreprocessor
from main.initialize import set_seed, add_code_search_args, get_fl_algorithm_initializer
from model.roberta_model import RobertaForSequenceClassification
from train.codesearch_fedrod_trainer import CodeSearchFedrodTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_code_search_args(parser)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(process)s %(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s',
        datefmt='%Y-%m-%d,%H:%M:%S')
    logging.info(args)

    set_seed(args.manual_seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config_class, model_class, tokenizer_class = RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer

    if args.do_train:
        # dataset attributes
        attributes = AbstractDataManager.load_attributes(args.data_file)
        num_labels = len(attributes["label_vocab"])
        args.num_labels = num_labels

        config = config_class.from_pretrained(args.model_name, num_labels=num_labels, finetuning_task='codesearch')
        tokenizer = tokenizer_class.from_pretrained(args.model_type)
        model = model_class.from_pretrained(args.model_name, config=config)
        model.to(device)

        # data
        preprocessor = CodeSearchPreprocessor(args=args, label_vocab=attributes["label_vocab"], tokenizer=tokenizer)
        manager = CodeSearchDataManager(args, preprocessor, args.data_type, args.data_file,
                                        args.train_batch_size, args.partition_file)

        train_loader_list, train_data_num_list, cls_num_list = manager.load_federated_data(server=False, count=True)
        args.cls_num_list = cls_num_list
        eval_data_loader = manager.load_path_data(args.eval_data_file)

        fl_algorithm = get_fl_algorithm_initializer(args.fl_algorithm)
        server_func = fl_algorithm(server=True)
        client_func = fl_algorithm(server=False)
        h_linear_state_list = [{} for _ in range(args.client_num_in_total)]
        for name, param in model.state_dict().item():
            if 'h_linear' in name:
                for i in range(args.client_num_in_total):
                    h_linear_state_list[i][name] = param.clone().detach().cpu()
        trainer = CodeSearchFedrodTrainer(args, device, model, h_linear_state_list=h_linear_state_list)

        clients = client_func(train_loader_list, train_data_num_list, None, device, args, trainer)
        server = server_func(clients, None, eval_data_loader, None, args, device, trainer)
        server.run()

        model.save_pretrained('cache/model/fedrod')
        tokenizer.save_pretrained('cache/model/fedrod')

    if args.do_test:
        config = config_class.from_pretrained(args.model_name, num_labels=2, finetuning_task='codesearch')
        tokenizer = tokenizer_class.from_pretrained(args.model_type)
        model = model_class.from_pretrained(args.model_name, config=config)
        model.to(device)

        preprocessor = CodeSearchPreprocessor(args=args, label_vocab=None, tokenizer=tokenizer)
        manager = CodeSearchDataManager(args, preprocessor, args.data_type, args.data_file,
                                        args.train_batch_size, args.partition_file)
        test_loader = manager.load_test_data()

        fl_algorithm = get_fl_algorithm_initializer(args.fl_algorithm)
        server_func = fl_algorithm(server=True)
        trainer = CodeSearchFedrodTrainer(args, device, model)
        server = server_func(None, None, test_loader, args, device, trainer)
        server.test()
