import argparse
import logging
import os

import torch
from transformers import RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer

from data.manager.codesearch_data_manager import CodeSearchDataManager
from data.preprocess.codesearch_preprocessor import CodeSearchPreprocessor
from main.initialize import set_seed, add_code_search_args, get_fl_algorithm_initializer
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

    config_class, model_class, tokenizer_class = RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer

    if args.do_train:

        config = config_class.from_pretrained(args.model_name, num_labels=2, finetuning_task='codesearch')
        tokenizer = tokenizer_class.from_pretrained(args.model_type)
        model = model_class.from_pretrained(args.model_name, config=config)
        model.to(device)

        # data
        preprocessor = CodeSearchPreprocessor(args, tokenizer)
        manager = CodeSearchDataManager(args, preprocessor)

        train_loader_list, train_data_num_list = manager.load_federated_data(False, 'train', args.train_data_file,
                                                                             args.train_batch_size,
                                                                             args.train_partition_file)

        fl_algorithm = get_fl_algorithm_initializer(args.fl_algorithm)
        server_func = fl_algorithm(server=True)
        client_func = fl_algorithm(server=False)

        trainer = CodeSearchTrainer(args, device, model)

        clients = client_func(train_loader_list, train_data_num_list, None, device, args, trainer)
        server = server_func(clients, None, None, args, device, trainer)
        server.run()

        save_dir = os.path.join(args.cache_dir, "model", args.fl_algorithm)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
