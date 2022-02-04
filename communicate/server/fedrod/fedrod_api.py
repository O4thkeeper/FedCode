from communicate.client.client_manager import ClientManager
from communicate.client.client_trainer import ClientTrainer
from communicate.server.fedrod.fedrod_aggregator import FedRodAggregator
from communicate.server.fedrod.fedrod_server_manager import FedRodServerManager


def fedrod_distributed(server):
    return init_server if server else init_client


def init_server(clients, train_loader, eval_loader, test_loader, args, device, trainer):
    trainer.set_data(train_loader, eval_loader, test_loader)
    aggregator = FedRodAggregator(trainer, args, device)

    server_manager = FedRodServerManager(aggregator, clients, args)
    return server_manager


def init_client(train_loader_list, train_data_num_list, test_loader_list, device, args, model_trainer):
    trainer = ClientTrainer(train_loader_list, train_data_num_list, test_loader_list, device, args, model_trainer)
    client_manager = ClientManager(args, trainer)
    return client_manager
