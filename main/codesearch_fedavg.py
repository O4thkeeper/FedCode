import argparse
import logging

import torch
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer

from data.manager.base.abstract_data_manager import AbstractDataManager
from data.manager.code_search_data_manager import CodeSearchDataManager
from data.preprocess.code_search_preprocessor import CodeSearchPreprocessor
from main.initialize import set_seed, add_federated_args, add_code_search_args, get_fl_algorithm_initializer
from model.model_args import ClassificationArgs
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

    # dataset attributes
    attributes = AbstractDataManager.load_attributes('data/store/codesearch/python_train.h5')
    num_labels = len(attributes["label_vocab"])

    model_args = ClassificationArgs()
    model_args.cache_dir = 'cache'
    model_args.model_class = 'codesearch'
    model_args.max_seq_length = 200
    model_args.model_type = 'classification'

    config_class, model_class, tokenizer_class = RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer
    config = config_class.from_pretrained('microsoft/codebert-base', num_labels=2, finetuning_task='codesearch')
    tokenizer = tokenizer_class.from_pretrained('roberta-base')
    model = model_class.from_pretrained('microsoft/codebert-base', config=config)
    model.to(device)

    # data
    preprocessor = CodeSearchPreprocessor(args=model_args, label_vocab=attributes["label_vocab"], tokenizer=tokenizer)
    mangager = CodeSearchDataManager(args, model_args, preprocessor, args.data_type, args.data_file,
                                     args.train_batch_size, args.partition_file)

    # loader = mangager.load_centralized_data()
    # load fed data
    train_loader_list, train_data_num_list = mangager.load_federated_data(server=False)

    fl_algorithm = get_fl_algorithm_initializer(args.fl_algorithm)
    server_func = fl_algorithm(server=True)
    client_func = fl_algorithm(server=False)

    trainer = CodeSearchTrainer(args, device, model)

    clients = client_func(train_loader_list, train_data_num_list, None, device, args, trainer)
    server = server_func(clients, None, None, args, device, trainer)
    server.run()

    # for step, batch in enumerate(loader):
    #     if step > 3:
    #         break
    #     logging.info('step %d' % step)
    #     logging.info(batch[0].shape)
    #     # logging.info(batch[1])
    #     # logging.info(batch[2])
    #     # logging.info(batch[3])
    #
    #
    # trainer.train()
    #
    model.save_pretrained('cache/model')
    tokenizer.save_pretrained('cache/model')
