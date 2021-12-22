import argparse
import logging
import torch

from data.manager.base.base_data_manager import BaseDataManager
from data.manager.text_classification_data_manager import TextClassificationDataManager
from data.preprocess.text_classification_preprocessor import TLMPreprocessor
from main.initialize import set_seed, create_model, add_federated_args, get_fl_algorithm_initializer
from model.model_args import ClassificationArgs
from train.fed_trainer_transformer import FedTransformerTrainer
from train.tc_transformer_trainer import TextClassificationTrainer

# sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "")))
# sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../")))
# sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
# sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../")))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = add_federated_args(parser)
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
    attributes = BaseDataManager.load_attributes(args.data_file_path)
    num_labels = len(attributes["label_vocab"])

    # create the model
    model_args = ClassificationArgs()
    model_args.model_name = args.model_name
    model_args.model_type = args.model_type
    model_args.load(model_args.model_name)
    model_args.num_labels = num_labels
    model_args.update_from_dict({"fl_algorithm": args.fl_algorithm,
                                 "freeze_layers": args.freeze_layers,
                                 "epochs": args.epochs,
                                 "learning_rate": args.lr,
                                 "gradient_accumulation_steps": args.gradient_accumulation_steps,
                                 "do_lower_case": args.do_lower_case,
                                 "manual_seed": args.manual_seed,
                                 # for ignoring the cache features.
                                 # "reprocess_input_data": args.reprocess_input_data,
                                 "overwrite_output_dir": True,
                                 "max_seq_length": args.max_seq_length,
                                 "train_batch_size": args.train_batch_size,
                                 "eval_batch_size": args.eval_batch_size,
                                 "evaluate_during_training": False,  # Disabled for FedAvg.
                                 "evaluate_during_training_steps": args.evaluate_during_training_steps,
                                 "fp16": args.fp16,
                                 "data_file_path": args.data_file_path,
                                 "partition_file_path": args.partition_file_path,
                                 "partition_method": args.partition_method,
                                 "dataset": args.dataset,
                                 "output_dir": args.output_dir,
                                 "is_debug_mode": args.is_debug_mode,
                                 "fedprox_mu": args.fedprox_mu
                                 })
    model_args.config["num_labels"] = num_labels

    model_config, client_model, tokenizer = create_model(model_args, formulation="classification")
    _, server_model, _ = create_model(model_args, formulation="classification")

    # trainer
    client_trainer = TextClassificationTrainer(model_args, device, client_model, None, None)
    client_fed_trainer = FedTransformerTrainer(client_trainer, client_model)

    server_trainer = TextClassificationTrainer(model_args, device, server_model, None, None)
    server_fed_trainer = FedTransformerTrainer(server_trainer, server_model)

    # data manager
    preprocessor = TLMPreprocessor(args=model_args, label_vocab=attributes["label_vocab"], tokenizer=tokenizer)
    dm = TextClassificationDataManager(args, model_args, preprocessor, num_workers=args.client_num_per_round)

    _, _, test_loader_server = dm.load_federated_data(server=True)
    train_loader_list, train_data_num_list, test_loader_list = dm.load_federated_data(server=False)

    fl_algorithm = get_fl_algorithm_initializer(args.fl_algorithm)
    server_func = fl_algorithm(server=True)
    client_func = fl_algorithm(server=False)

    clients = client_func(train_loader_list, train_data_num_list, test_loader_list, device, args, client_fed_trainer)
    server = server_func(clients, None, test_loader_server, args, device, server_fed_trainer)
    server.run()
