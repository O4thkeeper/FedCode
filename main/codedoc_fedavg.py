import argparse
import logging

import torch
from torch import nn
from transformers import RobertaConfig, RobertaTokenizer, RobertaModel

from data.manager.code_doc_data_manager import CodeDocDataManager
from data.preprocess.code_doc_preprocessor import CodeDocPreprocessor
from main.initialize import set_seed, get_fl_algorithm_initializer, add_code_doc_args
from model.seq2seq_model import Seq2Seq
from train.codedoc_trainer import CodeDocTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_code_doc_args(parser)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(process)s %(asctime)s.%(msecs)03d - {%(module)s.py (%(lineno)d)} - %(funcName)s(): %(message)s',
        datefmt='%Y-%m-%d,%H:%M:%S')
    logging.info(args)

    set_seed(args.manual_seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config_class, model_class, tokenizer_class = RobertaConfig, RobertaModel, RobertaTokenizer

    if args.do_train:
        config = config_class.from_pretrained(args.model_name)
        tokenizer = tokenizer_class.from_pretrained(args.model_name)

        encoder = model_class.from_pretrained(args.model_name, config=config)
        decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        model = Seq2Seq(encoder=encoder, decoder=decoder, config=config,
                        beam_size=args.beam_size, max_length=args.max_target_length,
                        sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)
        model.to(device)

        # data
        preprocessor = CodeDocPreprocessor(args=args, tokenizer=tokenizer)
        manager = CodeDocDataManager(args, preprocessor)

        train_loader_list, train_data_num_list = manager.load_federated_data(False, 'train', args.train_data_file,
                                                                             args.train_batch_size,
                                                                             args.train_partition_file)
        eval_loader_list, eval_data_num_list = manager.load_federated_data(False, 'eval', args.eval_data_file,
                                                                           args.eval_batch_size,
                                                                           args.eval_partition_file)
        test_loader_list, test_data_num_list = manager.load_federated_data(False, 'test', args.eval_data_file,
                                                                           args.eval_batch_size,
                                                                           args.eval_partition_file)

        fl_algorithm = get_fl_algorithm_initializer(args.fl_algorithm)
        server_func = fl_algorithm(server=True)
        client_func = fl_algorithm(server=False)

        trainer = CodeDocTrainer(args, device, model, tokenizer)

        clients = client_func(train_loader_list, train_data_num_list, test_loader_list, device, args, trainer,
                              eval_loader_list)
        server = server_func(clients, None, None, args, device, trainer)
        server.run()

        torch.save(model.state_dict(),'cache/model/codedoc_fedavg/model.pt')

    if args.do_test:
        pass
        # config = config_class.from_pretrained(args.model_name, num_labels=2, finetuning_task='codesearch')
        # tokenizer = tokenizer_class.from_pretrained(args.model_type)
        # model = model_class.from_pretrained(args.model_name, config=config)
        # model.to(device)
        #
        # preprocessor = CodeSearchPreprocessor(args=args, label_vocab=None, tokenizer=tokenizer)
        # manager = CodeSearchDataManager(args, preprocessor, args.data_type, args.data_file,
        #                                 args.train_batch_size, args.partition_file)
        # test_loader = manager.load_test_data()

        # fl_algorithm = get_fl_algorithm_initializer(args.fl_algorithm)
        # server_func = fl_algorithm(server=True)
        # trainer = CodeSearchTrainer(args, device, model)
        # server = server_func(None, None, test_loader, args, device, trainer)
        # server.test()
