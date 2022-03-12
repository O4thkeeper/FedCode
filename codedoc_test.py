import argparse
import logging
import os.path

import torch
from torch import nn
from torch.utils.data import RandomSampler
from transformers import RobertaConfig, RobertaTokenizer, RobertaModel

from data.manager.codedoc_data_manager import CodeDocDataManager
from data.preprocess.base.base_data_loader import BaseDataLoader
from data.preprocess.codedoc_preprocessor import CodeDocPreprocessor
from main.initialize import set_seed, get_fl_algorithm_initializer, add_code_doc_args
from model.seq2seq_model import Seq2Seq

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

    model.to(device)

    state, res = manager._load_data_loader_from_cache(1,'train')
    examples, features, dataset = res
    sampler = RandomSampler(dataset)
    data_loader = BaseDataLoader(examples, features, dataset,
                                 sampler=sampler,
                                 batch_size=64)
    test_loader = manager.load_federated_data(True, 'test', args.eval_data_file, args.eval_batch_size,
                                              max_size=1000)

    for i, batch in enumerate(data_loader):
        if i > 0:
            break
        batch = tuple(t.to(device) for t in batch)
        source_ids, source_mask, target_ids, target_mask = batch
        logging.info("source_ids %s" % source_ids)
        logging.info("source_mask %s" % source_mask)
        logging.info("target_ids %s" % target_ids)
        logging.info("target_mask %s" % target_mask)

        outputs = model.encoder(source_ids, attention_mask=source_mask)
        logging.info("outputs shape %s" % outputs.shape)

        encoder_output = outputs[0].permute([1, 0, 2]).contiguous()
        logging.info("encoder_output shape %s" % encoder_output.shape)

        attn_mask = -1e4 * (1 - model.bias[:target_ids.shape[1], :target_ids.shape[1]])
        logging.info("attn_mask shape %s" % attn_mask.shape)

        tgt_embeddings = model.encoder.embeddings(target_ids).permute([1, 0, 2]).contiguous()
        logging.info("tgt_embeddings shape %s" % tgt_embeddings.shape)

        out = model.decoder(tgt_embeddings, encoder_output, tgt_mask=attn_mask,
                            memory_key_padding_mask=(1 - source_mask).bool())
        logging.info("out shape %s" % out.shape)

        hidden_states = torch.tanh(model.dense(out)).permute([1, 0, 2]).contiguous()
        logging.info("hidden_states shape %s" % hidden_states.shape)

        lm_logits = model.lm_head(hidden_states)
        logging.info("lm_logits shape %s" % lm_logits.shape)

        # Shift so that tokens < n predict n
        active_loss = target_mask[..., 1:].ne(0).view(-1) == 1
        logging.info("active_loss shape %s" % active_loss.shape)

        shift_logits = lm_logits[..., :-1, :].contiguous()
        logging.info("shift_logits shape %s" % shift_logits.shape)

        shift_labels = target_ids[..., 1:].contiguous()
        logging.info("shift_labels shape %s" % shift_labels.shape)

        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        global_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                               shift_labels.view(-1)[active_loss])

        outputs = global_loss, global_loss * active_loss.sum(), active_loss.sum()
        logging.info(outputs)
