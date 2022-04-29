import argparse
import logging
import os.path

import torch
from torch import nn
from transformers import RobertaConfig, RobertaTokenizer, RobertaModel

from data.manager.codedoc_data_manager import CodeDocDataManager
from data.preprocess.codedoc_preprocessor import CodeDocPreprocessor
from main.initialize import set_seed, get_fl_algorithm_initializer, add_code_doc_args
from model.seq2seq_model import Seq2Seq
from test.codedoc_trainer_for_test import CodeDocTrainer

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

    config = config_class.from_pretrained(args.model_name)
    tokenizer = tokenizer_class.from_pretrained(args.model_name)

    encoder = model_class.from_pretrained(args.model_name, config=config)
    decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model = Seq2Seq(encoder=encoder, decoder=decoder, config=config,
                    beam_size=args.beam_size, max_length=args.max_target_length,
                    sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)

    preprocessor = CodeDocPreprocessor(args=args, tokenizer=tokenizer)
    manager = CodeDocDataManager(args, preprocessor)

    if args.do_train:
        pass

    if args.do_test:
        model.load_state_dict(torch.load(args.load_model))
        model.to(device)

        test_loader = manager.load_federated_data(True, 'test', args.test_data_file, args.eval_batch_size)

        fl_algorithm = get_fl_algorithm_initializer(args.fl_algorithm)
        server_func = fl_algorithm(server=True)

        trainer = CodeDocTrainer(args, device, model, tokenizer)

        server = server_func(None, None, test_loader, args, device, trainer, None)
        server.test()