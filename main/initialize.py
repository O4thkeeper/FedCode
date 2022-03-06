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
        raise Exception("no such fed algorithm")

    return fl_algorithm


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def  add_code_search_args(parser):

    parser.add_argument('--dataset', type=str, default='codesearch')

    parser.add_argument('--language', type=str, default='python')

    parser.add_argument('--train_data_file', type=str, default='')

    parser.add_argument('--train_partition_file', type=str, default='')

    parser.add_argument('--eval_data_file', type=str, default='')

    parser.add_argument('--eval_partition_file', type=str, default='')

    parser.add_argument('--test_data_file', type=str, default='')

    parser.add_argument('--test_partition_file', type=str, default='')

    parser.add_argument('--server_data', type=str, default='')

    parser.add_argument('--partition_method', type=str)

    parser.add_argument('--model_type', type=str, default='roberta-base')

    parser.add_argument('--model_name', type=str, default='microsoft/codebert-base')

    parser.add_argument('--do_lower_case', type=bool, default=True)

    parser.add_argument('--train_batch_size', type=int, default=64)

    parser.add_argument('--eval_batch_size', type=int, default=32)

    parser.add_argument('--max_seq_length', type=int, default=200)

    parser.add_argument('--fp16', action="store_true")

    parser.add_argument('--manual_seed', type=int, default=42)

    parser.add_argument('--output_dir', type=str, default="tmp/")

    parser.add_argument('--fl_algorithm', type=str, default="FedAvg")

    parser.add_argument('--comm_round', type=int, default=9)

    parser.add_argument('--client_num_in_total', type=int, default=15)

    parser.add_argument('--client_num_per_round', type=int,default=5)

    parser.add_argument('--epochs', type=int, default=1)

    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)

    parser.add_argument('--client_optimizer', type=str, default='adam')

    parser.add_argument('--learning_rate', type=float, default=5e-5)

    parser.add_argument('--max_grad_norm', type=float, default=1.0)

    parser.add_argument('--adam_epsilon', type=float, default=1e-8)

    parser.add_argument('--warmup_ratio', type=float, default=0.06)

    parser.add_argument('--weight_decay', type=float, default=0)

    parser.add_argument('--server_optimizer', type=str, default='sgd')

    parser.add_argument('--server_lr', type=float, default=0.1)

    parser.add_argument('--cache_dir', type=str, default="cache/codesearch/")

    parser.add_argument('--freeze_layers', type=str, default='')

    parser.add_argument('--do_train', action="store_true")

    parser.add_argument('--do_eval', action="store_true")

    parser.add_argument('--do_test', action="store_true")

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

    parser.add_argument('--language', type=str, default='python')

    parser.add_argument('--train_data_file', type=str, default='')

    parser.add_argument('--train_partition_file', type=str, default='')

    parser.add_argument('--eval_data_file', type=str, default='')

    parser.add_argument('--eval_partition_file', type=str, default='')

    parser.add_argument('--test_data_file', type=str, default='')

    parser.add_argument('--test_partition_file', type=str, default='')

    parser.add_argument('--server_data', type=str, default='')

    parser.add_argument('--partition_method', type=str)

    parser.add_argument('--model_type', type=str, default='roberta-base')

    parser.add_argument('--model_name', type=str, default='microsoft/codebert-base')

    parser.add_argument('--load_model',type=str,default='')

    parser.add_argument('--do_lower_case', type=bool, default=True)

    parser.add_argument('--train_batch_size', type=int, default=64)

    parser.add_argument('--eval_batch_size', type=int, default=32)

    parser.add_argument('--max_seq_length', type=int, default=256)

    parser.add_argument('--max_target_length', type=int, default=128)

    parser.add_argument('--fp16', action="store_true")

    parser.add_argument('--manual_seed', type=int, default=42)

    parser.add_argument('--output_dir', type=str, default="tmp/")

    parser.add_argument('--fl_algorithm', type=str, default="FedAvg")

    parser.add_argument('--comm_round', type=int, default=10)

    parser.add_argument('--client_num_in_total', type=int, default=-1)

    parser.add_argument('--client_num_per_round', type=int, default=5)

    parser.add_argument('--epochs', type=int, default=1)

    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)

    parser.add_argument('--client_optimizer', type=str, default='adam')

    parser.add_argument('--learning_rate', type=float, default=5e-5)

    parser.add_argument('--max_grad_norm', type=float, default=1.0)

    parser.add_argument("--beam_size", default=10, type=int)

    parser.add_argument('--adam_epsilon', type=float, default=1e-8)

    parser.add_argument('--warmup_ratio', type=float, default=0.06)

    parser.add_argument('--weight_decay', type=float, default=0)

    parser.add_argument('--server_optimizer', type=str, default='sgd')

    parser.add_argument('--server_lr', type=float, default=0.1)

    parser.add_argument('--server_local_steps', type=int, default=1)

    parser.add_argument('--cache_dir', type=str, default="cache/")

    parser.add_argument('--do_train', action="store_true")

    parser.add_argument('--do_eval', action="store_true")

    parser.add_argument('--do_test', action="store_true")

    parser.add_argument('--test_mode', type=str, default='acc')

    return parser
