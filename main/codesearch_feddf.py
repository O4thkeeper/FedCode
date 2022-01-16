import argparse
import logging

import torch
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer

from data.manager.base.abstract_data_manager import AbstractDataManager
from data.manager.code_search_data_manager import CodeSearchDataManager
from data.preprocess.code_search_preprocessor import CodeSearchPreprocessor
from main.initialize import add_code_search_args, set_seed, get_fl_algorithm_initializer
from train.codesearch_trainer import CodeSearchTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_code_search_args(parser)
    args = parser.parse_args()

    # customize the log format
    logging.basicConfig(
        level=logging.INFO,
        format='%(process)s %(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s',
        datefmt='%Y-%m-%d,%H:%M:%S')
    logging.info(args)

    set_seed(args.manual_seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device

    config_class, model_class, tokenizer_class = RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer

    if args.do_train:
        # dataset attributes
        attributes = AbstractDataManager.load_attributes(args.data_file)
        num_labels = len(attributes["label_vocab"])

        config = config_class.from_pretrained(args.model_name, num_labels=num_labels, finetuning_task='codesearch')
        tokenizer = tokenizer_class.from_pretrained(args.model_type)
        client_model = model_class.from_pretrained(args.model_name, config=config)
        server_model = model_class.from_pretrained(args.model_name, config=config)

        # data
        preprocessor = CodeSearchPreprocessor(args=args, label_vocab=attributes["label_vocab"], tokenizer=tokenizer)
        manager = CodeSearchDataManager(args, preprocessor, args.data_type, args.data_file,
                                        args.train_batch_size, args.partition_file)

        train_loader_list, train_data_num_list = manager.load_federated_data(server=False)
        server_data_loader = manager.load_path_data("data/store/codesearch/train_valid/python/valid1.txt")

        fl_algorithm = get_fl_algorithm_initializer(args.fl_algorithm)  # "FedDf"
        server_func = fl_algorithm(server=True)
        client_func = fl_algorithm(server=False)

        client_trainer = CodeSearchTrainer(args, device, client_model)
        server_trainer = CodeSearchTrainer(args, device, server_model)

        clients = client_func(train_loader_list, train_data_num_list, None, device, args, client_trainer)
        server = server_func(clients, None, None, server_data_loader, args, device, server_trainer, client_trainer)
        server.run()

        server_model.save_pretrained('cache/model/codesearch_feddf')
        tokenizer.save_pretrained('cache/model/codesearch_feddf')

    # if args.do_test:
    #     # config = config_class.from_pretrained('cache/model/config.json', num_labels=num_labels,
    #     #                                       finetuning_task='codesearch')
    #     # tokenizer = tokenizer_class.from_pretrained('roberta-base')
    #     # model = model_class.from_pretrained('cache/model/pytorch_model.bin', config=config)
    #     config = config_class.from_pretrained(args.model_name, num_labels=2, finetuning_task='codesearch')
    #     tokenizer = tokenizer_class.from_pretrained(args.model_type)
    #     model = model_class.from_pretrained(args.model_name, config=config)
    #     model.to(device)
    #
    #     preprocessor = CodeSearchPreprocessor(args=args, label_vocab=None, tokenizer=tokenizer)
    #     manager = CodeSearchDataManager(args, preprocessor, args.data_type, args.data_file,
    #                                     args.train_batch_size, args.partition_file)
    #     test_loader = manager.load_test_data()
    #
    #     fl_algorithm = get_fl_algorithm_initializer(args.fl_algorithm)
    #     server_func = fl_algorithm(server=True)
    #     trainer = CodeSearchTrainer(args, device, model)
    #     server = server_func(None, None, test_loader, args, device, trainer)
    #     server.test()
