from communicate.client.client_manager import ClientManager
from communicate.client.client_trainer import ClientTrainer
from communicate.server.feddf.feddf_aggregator import FedDfAggregator
from communicate.server.feddf.feddf_server_manager import FedDfServerManager


def feddf_distributed(server):
    return init_server if server else init_client


def init_server(clients, train_loader, test_loader, server_data_loader, args, device, server_trainer, client_trainer):
    client_trainer.set_data(train_loader, None, test_loader)
    aggregator = FedDfAggregator(server_trainer, client_trainer, server_data_loader, args, device)

    server_manager = FedDfServerManager(aggregator, clients, args)
    return server_manager


def init_client(train_loader_list, train_data_num_list, test_loader_list, device, args, model_trainer):
    trainer = ClientTrainer(train_loader_list, train_data_num_list, test_loader_list, device, args, model_trainer)
    client_manager = ClientManager(args, trainer)
    return client_manager
