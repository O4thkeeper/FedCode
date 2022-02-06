from communicate.server.fedavg.fedavg_aggregator import FedAVGAggregator
from communicate.client.client_manager import ClientManager
from communicate.server.fedavg.fedavg_server_manager import FedAVGServerManager
from communicate.client.client_trainer import ClientTrainer


def fedAvg_distributed(server):
    return init_server if server else init_client


def init_server(clients, train_loader, test_loader, args, device, model_trainer):
    model_trainer.set_data(train_loader, None, test_loader)
    aggregator = FedAVGAggregator(model_trainer, args, device)

    server_manager = FedAVGServerManager(aggregator, clients, args)
    return server_manager


def init_client(train_loader_list, train_data_num_list, test_loader_list, device, args, model_trainer,
                eval_loader_list=None):
    trainer = ClientTrainer(train_loader_list, train_data_num_list, test_loader_list, device, args, model_trainer,
                            eval_loader_list)
    client_manager = ClientManager(args, trainer)
    return client_manager
