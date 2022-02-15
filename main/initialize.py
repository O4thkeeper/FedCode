import random

import numpy as np
import torch
from transformers import (
    BertConfig,
    BertTokenizer,
    BertForTokenClassification,
    BertForQuestionAnswering,
    DistilBertConfig,
    DistilBertTokenizer,
    DistilBertForTokenClassification,
    DistilBertForQuestionAnswering,
    BartConfig,
    BartForConditionalGeneration,
    BartTokenizer,
)

from communicate.server.fedavg.fedavg_api import fedAvg_distributed
from communicate.server.feddf.feddf_api import feddf_distributed
from communicate.server.fedrod.fedrod_api import fedrod_distributed
from model.bert_model import BertForSequenceClassification
from model.distilbert_model import DistilBertForSequenceClassification


def get_fl_algorithm_initializer(alg_name):
    if alg_name == "FedAvg":
        fl_algorithm = fedAvg_distributed
    elif alg_name == "FedDf":
        fl_algorithm = feddf_distributed
    elif alg_name == "FedRod":
        fl_algorithm = fedrod_distributed
    else:
        raise Exception("please do sanity check for this algorithm.")

    return fl_algorithm


def create_model(args, formulation="classification"):
    # create model, tokenizer, and model config (HuggingFace style)
    MODEL_CLASSES = {
        "classification": {
            "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
            "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
            # "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
            # "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
        },
        "seq_tagging": {
            "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
            "distilbert": (DistilBertConfig, DistilBertForTokenClassification, DistilBertTokenizer),
        },
        "span_extraction": {
            "bert": (BertConfig, BertForQuestionAnswering, BertTokenizer),
            "distilbert": (DistilBertConfig, DistilBertForQuestionAnswering, DistilBertTokenizer),
        },
        "seq2seq": {
            "bart": (BartConfig, BartForConditionalGeneration, BartTokenizer),
        }
    }
    config_class, model_class, tokenizer_class = MODEL_CLASSES[formulation][args.model_type]
    config = config_class.from_pretrained(args.model_name, **args.config)
    model = model_class.from_pretrained(args.model_name, config=config)
    if formulation != "seq2seq":
        tokenizer = tokenizer_class.from_pretrained(
            args.model_name, do_lower_case=args.do_lower_case)
    else:
        tokenizer = [None, None]
        tokenizer[0] = tokenizer_class.from_pretrained(args.model_name)
        tokenizer[1] = tokenizer[0]
    return config, model, tokenizer


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def add_federated_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """

    parser.add_argument("--is_debug_mode", default=0, type=int,
                        help="is_debug_mode")

    # Data related
    parser.add_argument('--dataset', type=str, default='agnews', metavar='N', help='dataset used for training')

    parser.add_argument('--data_file_path', type=str,
                        default='/home/bill/fednlp_data/data_files/agnews_data.h5',
                        help='data h5 file path')

    parser.add_argument('--partition_file_path', type=str,
                        default='/home/bill/fednlp_data/partition_files/agnews_partition.h5',
                        help='partition h5 file path')

    parser.add_argument('--partition_method', type=str, default='uniform',
                        help='partition method')

    # Model related
    parser.add_argument('--model_type', type=str, default='bert', metavar='N',
                        help='transformer model type')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased', metavar='N',
                        help='transformer model name')
    parser.add_argument('--do_lower_case', type=bool, default=True, metavar='N',
                        help='transformer model name')

    # Learning related
    parser.add_argument('--train_batch_size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 8)')
    parser.add_argument('--eval_batch_size', type=int, default=8, metavar='N',
                        help='input batch size for evaluation (default: 8)')

    parser.add_argument('--max_seq_length', type=int, default=128, metavar='N',
                        help='maximum sequence length (default: 128)')

    parser.add_argument('--fp16', default=False, action="store_true",
                        help='if enable fp16 for training')

    parser.add_argument('--manual_seed', type=int, default=42, metavar='N',
                        help='random seed')

    # IO related
    parser.add_argument('--output_dir', type=str, default="tmp/", metavar='N',
                        help='path to save the trained results and ckpts')

    # Federated Learning related
    parser.add_argument('--fl_algorithm', type=str, default="FedAvg",
                        help='Algorithm list: FedAvg; FedOPT; FedProx ')

    parser.add_argument('--comm_round', type=int, default=10,
                        help='how many round of communications we shoud use')

    parser.add_argument('--client_num_in_total', type=int, default=-1, metavar='NN',
                        help='number of clients in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int,
                        default=4, metavar='NN', help='number of workers')

    parser.add_argument('--epochs', type=int, default=3, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, metavar='EP',
                        help='how many steps for accumulate the loss.')

    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='Optimizer used on the client. This field can be the name of any subclass of the torch Opimizer class.')

    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate on the client (default: 0.001)')

    parser.add_argument('--weight_decay', type=float, default=0, metavar='N',
                        help='L2 penalty')

    parser.add_argument('--server_optimizer', type=str, default='sgd',
                        help='Optimizer used on the server. This field can be the name of any subclass of the torch Opimizer class.')

    parser.add_argument('--server_lr', type=float, default=0.1,
                        help='server learning rate (default: 0.001)')

    parser.add_argument('--server_momentum', type=float, default=0,
                        help='server momentum (default: 0)')

    parser.add_argument('--fedprox_mu', type=float, default=1,
                        help='server momentum (default: 1)')

    parser.add_argument(
        '--evaluate_during_training_steps', type=int, default=100, metavar='EP',
        help='the frequency of the evaluation during training')

    parser.add_argument('--frequency_of_the_test', type=int, default=1,
                        help='the frequency of the algorithms')

    # cached related
    parser.add_argument('--cache_dir', type=str, default="cache/", metavar='N',
                        help='cache file path')

    # freeze related
    parser.add_argument('--freeze_layers', type=str, default='', metavar='N',
                        help='freeze which layers')

    return parser


def add_code_search_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """

    parser.add_argument("--is_debug_mode", default=0, type=int,
                        help="is_debug_mode")

    # Data related
    parser.add_argument('--dataset', type=str, default='agnews', metavar='N', help='dataset used for training')

    parser.add_argument('--data_file', type=str,
                        default='data/store/codesearch/python_train.h5',
                        help='data h5 file path')

    parser.add_argument('--partition_file', type=str,
                        default='data/store/codesearch/python_train_partition.h5',
                        help='partition h5 file path')

    parser.add_argument('--server_data', type=str, default='')

    parser.add_argument('--eval_data_file', type=str, default='')

    parser.add_argument('--partition_method', type=str, help='partition method')

    parser.add_argument('--data_type', type=str, default='train', help='train or test')

    # Model related
    parser.add_argument('--model_type', type=str, default='roberta-base', metavar='N',
                        help='transformer model type')

    parser.add_argument('--model_name', type=str, default='microsoft/codebert-base', metavar='N',
                        help='transformer model name')

    parser.add_argument('--do_lower_case', type=bool, default=True, metavar='N',
                        help='transformer model name')

    # Learning related
    parser.add_argument('--train_batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 8)')

    parser.add_argument('--eval_batch_size', type=int, default=32, metavar='N',
                        help='input batch size for evaluation (default: 8)')

    parser.add_argument('--max_seq_length', type=int, default=200, metavar='N',
                        help='maximum sequence length (default: 200)')

    parser.add_argument('--fp16', action="store_true", help='if enable fp16 for training')

    parser.add_argument('--manual_seed', type=int, default=42, metavar='N', help='random seed')

    # IO related
    parser.add_argument('--output_dir', type=str, default="tmp/", metavar='N',
                        help='path to save the trained results and ckpts')

    # Federated Learning related
    parser.add_argument('--fl_algorithm', type=str, default="FedAvg",
                        help='Algorithm list: FedAvg; FedOPT; FedProx ')

    parser.add_argument('--comm_round', type=int, default=10,
                        help='how many round of communications we shoud use')

    parser.add_argument('--client_num_in_total', type=int, default=-1, metavar='NN',
                        help='number of clients in a distributed cluster')

    parser.add_argument('--client_num_per_round', type=int,
                        default=4, metavar='NN', help='number of workers')

    parser.add_argument('--epochs', type=int, default=3, metavar='EP',
                        help='how many epochs will be trained locally')

    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, metavar='EP',
                        help='how many steps for accumulate the loss.')

    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='Optimizer used on the client. This field can be the name of any subclass of the torch Opimizer class.')

    parser.add_argument('--learning_rate', type=float, default=5e-5, metavar='LR',
                        help='learning rate on the client (default: 0.001)')

    parser.add_argument('--max_grad_norm', type=float, default=1.0, metavar='MGN',
                        help='learning rate on the client (default: 0.001)')

    parser.add_argument('--adam_epsilon', type=float, default=1e-8, metavar='AE',
                        help='')

    parser.add_argument('--warmup_ratio', type=float, default=0.06, metavar='WR',
                        help='')

    parser.add_argument('--weight_decay', type=float, default=0, metavar='WD',
                        help='L2 penalty')

    parser.add_argument('--server_optimizer', type=str, default='sgd',
                        help='Optimizer used on the server. This field can be the name of any subclass of the torch Opimizer class.')

    parser.add_argument('--server_lr', type=float, default=0.1)

    parser.add_argument('--server_local_steps', type=int, default=1)

    parser.add_argument('--server_momentum', type=float, default=0,
                        help='server momentum (default: 0)')

    parser.add_argument('--fedprox_mu', type=float, default=1,
                        help='server momentum (default: 1)')

    parser.add_argument(
        '--evaluate_during_training_steps', type=int, default=100, metavar='EP',
        help='the frequency of the evaluation during training')

    parser.add_argument('--frequency_of_the_test', type=int, default=1,
                        help='the frequency of the algorithms')

    # cached related
    parser.add_argument('--cache_dir', type=str, default="cache/", metavar='N',
                        help='cache file path')

    # freeze related
    parser.add_argument('--freeze_layers', type=str, default='', metavar='N',
                        help='freeze which layers')

    # mode related
    parser.add_argument('--do_train', action="store_true")

    parser.add_argument('--do_eval', action="store_true")

    parser.add_argument('--do_test', action="store_true")

    parser.add_argument('--test_mode', type=str, default='acc')

    return parser


def add_mrr_test_args(parser):
    parser.add_argument('--model_type', type=str, default='roberta-base')

    parser.add_argument('--model_name', type=str, default='microsoft/codebert-base')

    parser.add_argument('--data_file', type=str, default='')

    parser.add_argument('--data_type', type=str, default='')

    parser.add_argument('--batch_size', type=int, default=64)

    parser.add_argument('--test_batch_size', type=int, default=1000)

    parser.add_argument('--output_dir', type=str, default="tmp/")

    parser.add_argument('--manual_seed', type=int, default=42)

    parser.add_argument('--max_seq_length', type=int, default=200)

    return parser


def add_code_doc_args(parser):
    parser.add_argument('--dataset', type=str, default='')

    parser.add_argument('--train_data_file', type=str, default='')

    parser.add_argument('--train_partition_file', type=str, default='')

    parser.add_argument('--eval_data_file', type=str, default='')

    parser.add_argument('--eval_partition_file', type=str, default='')

    parser.add_argument('--test_data_file', type=str, default='')

    parser.add_argument('--test_partition_file', type=str, default='')

    parser.add_argument('--server_data', type=str, default='')

    parser.add_argument('--partition_method', type=str, help='partition method')

    parser.add_argument('--model_type', type=str, default='roberta-base', metavar='N',
                        help='transformer model type')

    parser.add_argument('--model_name', type=str, default='microsoft/codebert-base', metavar='N',
                        help='transformer model name')

    parser.add_argument('--do_lower_case', type=bool, default=True, metavar='N',
                        help='transformer model name')

    parser.add_argument('--train_batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 8)')

    parser.add_argument('--eval_batch_size', type=int, default=32, metavar='N',
                        help='input batch size for evaluation (default: 8)')

    parser.add_argument('--max_seq_length', type=int, default=256)

    parser.add_argument('--max_target_length', type=int, default=128)

    parser.add_argument('--fp16', action="store_true", help='if enable fp16 for training')

    parser.add_argument('--manual_seed', type=int, default=42, metavar='N', help='random seed')

    parser.add_argument('--output_dir', type=str, default="tmp/", metavar='N',
                        help='path to save the trained results and ckpts')

    parser.add_argument('--fl_algorithm', type=str, default="FedAvg",
                        help='Algorithm list: FedAvg; FedOPT; FedProx ')

    parser.add_argument('--comm_round', type=int, default=10)

    parser.add_argument('--client_num_in_total', type=int, default=-1)

    parser.add_argument('--client_num_per_round', type=int, default=5)

    parser.add_argument('--epochs', type=int, default=1)

    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, metavar='EP',
                        help='how many steps for accumulate the loss.')

    parser.add_argument('--client_optimizer', type=str, default='adam',
                        help='Optimizer used on the client. This field can be the name of any subclass of the torch Opimizer class.')

    parser.add_argument('--learning_rate', type=float, default=5e-5, metavar='LR',
                        help='learning rate on the client (default: 0.001)')

    parser.add_argument('--max_grad_norm', type=float, default=1.0, metavar='MGN',
                        help='learning rate on the client (default: 0.001)')

    parser.add_argument("--beam_size", default=10, type=int, help="beam size for beam search")

    parser.add_argument('--adam_epsilon', type=float, default=1e-8, metavar='AE',
                        help='')

    parser.add_argument('--warmup_ratio', type=float, default=0.06, metavar='WR',
                        help='')

    parser.add_argument('--weight_decay', type=float, default=0, metavar='WD',
                        help='L2 penalty')

    parser.add_argument('--server_optimizer', type=str, default='sgd',
                        help='Optimizer used on the server. This field can be the name of any subclass of the torch Opimizer class.')

    parser.add_argument('--server_lr', type=float, default=0.1)

    parser.add_argument('--server_local_steps', type=int, default=1)

    parser.add_argument('--cache_dir', type=str, default="cache/")

    parser.add_argument('--do_train', action="store_true")

    parser.add_argument('--do_eval', action="store_true")

    parser.add_argument('--do_test', action="store_true")

    parser.add_argument('--test_mode', type=str, default='acc')

    return parser
