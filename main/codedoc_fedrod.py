import argparse
import logging
import os.path
import time

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from transformers import RobertaConfig, RobertaTokenizer, RobertaModel

from data.manager.codedoc_data_manager import CodeDocDataManager
from data.preprocess.codedoc_preprocessor import CodeDocPreprocessor
from main.initialize import set_seed, get_fl_algorithm_initializer, add_code_doc_args
from model.biseq2seq_model import Seq2Seq
from train.codedoc_fedrod_trainer import CodeDocFedRodTrainer

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
                    sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id, p_head=True)

    preprocessor = CodeDocPreprocessor(args=args, tokenizer=tokenizer)
    manager = CodeDocDataManager(args, preprocessor)

    if args.do_train:
        model.to(device)

        train_loader_list, train_data_num_list, label_num_list = manager.load_federated_data(False, 'train',
                                                                                             args.train_data_file,
                                                                                             args.train_batch_size,
                                                                                             args.train_partition_file)
        eval_loader = manager.load_federated_data(True, 'eval', args.eval_data_file, args.eval_batch_size)
        test_loader = manager.load_federated_data(True, 'test', args.eval_data_file, args.eval_batch_size,
                                                  max_size=1000)

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

        # counter_list = [Counter() for _ in range(args.client_num_in_total)]
        # vocab_weight_list = [[0 for _ in range(config.vocab_size)] for _ in range(args.client_num_in_total)]
        # for loader, counter, vocab_weight in tqdm(zip(train_loader_list, counter_list, vocab_weight_list),
        #                                           desc="counting vocab"):
        #     for batch in loader:
        #         source_ids, _, target_ids, _ = batch
        #         for i in range(source_ids.shape[0]):
        #             counter.update(source_ids[i].tolist())
        #             counter.update(target_ids[i].tolist())
        #     for key, value in counter.items():
        #         vocab_weight[int(key)] = value
        # vocab_weight_list = torch.Tensor(vocab_weight_list)
        # vocab_weight_list = vocab_weight_list / vocab_weight_list.sum(dim=1).view(-1, 1)

        trainer = CodeDocFedRodTrainer(args, device, model, tokenizer, p_head_state_list=p_head_state_list)

        clients = client_func(train_loader_list, train_data_num_list, None, device, args, trainer)
        server = server_func(clients, None, eval_loader, test_loader, args, device, trainer)
        server.run()

        save_dir = os.path.join(args.cache_dir, "model", args.fl_algorithm)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))
        torch.save(trainer.p_head_state_list, os.path.join(save_dir, 'p_head.pt'))
        # torch.save(vocab_weight_list, os.path.join(save_dir, 'vocab.pt'))
        torch.save(args.label_weight, os.path.join(save_dir, "label_weight.pt"))

    if args.do_test:
        model.load_state_dict(torch.load(os.path.join(args.load_model, 'model.pt')))
        model.to(device)

        test_loader = manager.load_federated_data(True, 'test', args.test_data_file, args.eval_batch_size)

        p_head_state_list = torch.load(os.path.join(args.load_model, 'p_head.pt'))
        # vocab_weight_list = torch.load(os.path.join(args.load_model, 'vocab.pt'))
        label_weight_list = torch.load(os.path.join(args.load_model, 'label_weight.pt'))

        trainer = CodeDocFedRodTrainer(args, device, model, tokenizer, test_dl=test_loader,
                                       p_head_state_list=p_head_state_list, label_weight_list=label_weight_list)
        g_bleu = trainer.test()
        with open(os.path.join(args.output_dir, 'fedrod_bleu_test_result.txt'), 'a') as f:
            f.write("TEST TIME:%s\n" % time.asctime(time.localtime(time.time())))
            f.write("global bleu-4: %s\n\n" % g_bleu)
        bleu_list = []
        for i in range(len(p_head_state_list)):
            trainer.set_model_params(trainer.get_model_params(), i)
            l_bleu = trainer.test(i)
            bleu_list.append(l_bleu)
        with open(os.path.join(args.output_dir, 'fedrod_bleu_test_result.txt'), 'a') as f:
            for i, bleu in enumerate(bleu_list):
                f.write("client %s bleu-4: %s\n\n" % (i, bleu))
            f.write("avg bleu-4: %s\n\n" % np.mean(bleu_list))
            f.write("max bleu-4: %s\n\n" % np.max(bleu_list))
            f.write("min bleu-4: %s\n\n" % np.min(bleu_list))
