import argparse
import logging
import os
import pickle

import torch
from transformers import RobertaConfig, RobertaTokenizer

from data.manager.codesearch_data_manager import CodeSearchDataManager
from data.preprocess.codesearch_preprocessor import CodeSearchPreprocessor
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
        args.num_labels = 2
        config = config_class.from_pretrained(args.model_name, num_labels=2, finetuning_task='codesearch')
        tokenizer = tokenizer_class.from_pretrained(args.model_type)
        model = model_class.from_pretrained(args.model_name, config=config, label_count=args.label_count)
        model.to(device)

        # data
        preprocessor = CodeSearchPreprocessor(args, tokenizer)
        manager = CodeSearchDataManager(args, preprocessor)

        train_loader_list, train_data_num_list, label_num_list = manager.load_federated_data(False, 'train',
                                                                                             args.train_data_file,
                                                                                             args.train_batch_size,
                                                                                             args.train_partition_file)

        args.cls_num_list = [torch.Tensor(cls_num) for cls_num in label_num_list]
        args.label_weight = [torch.Tensor(cls_num) / sum(cls_num) for cls_num in label_num_list]

        fl_algorithm = get_fl_algorithm_initializer(args.fl_algorithm)
        server_func = fl_algorithm(server=True)
        client_func = fl_algorithm(server=False)
        p_head_state_list = [{} for _ in range(args.client_num_in_total)]
        for name, param in model.state_dict().items():
            if 'p_head' in name:
                for i in range(args.client_num_in_total):
                    p_head_state_list[i][name] = param.clone().detach().cpu()
        trainer = CodeSearchFedrodTrainer(args, device, model, p_head_state_list=p_head_state_list)

        clients = client_func(train_loader_list, train_data_num_list, None, device, args, trainer)
        server = server_func(clients, None, None, None, args, device, trainer)
        server.run()

        save_dir = os.path.join(args.cache_dir, "model", args.fl_algorithm)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
        torch.save(trainer.p_head_state_list, os.path.join(save_dir, "p_head.pt"))
        torch.save(args.label_weight, os.path.join(save_dir, "label_weight.pt"))
