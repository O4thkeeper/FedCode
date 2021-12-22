from communicate.api.fedavg_aggregator import FedAVGAggregator
from communicate.api.fedavg_client_manager import FedAVGClientManager
from communicate.api.fedavg_server_manager import FedAVGServerManager
from communicate.api.fedavg_trainer import FedAVGTrainer


def FedML_FedAvg_distributed(server):
    return init_server if server else init_client


def init_server(clients, train_loader, test_loader, args, device, model_trainer):
    model_trainer.model_trainer.set_data(train_loader, test_loader)
    aggregator = FedAVGAggregator(model_trainer, args, device)

    server_manager = FedAVGServerManager(aggregator, clients, args)
    return server_manager


def init_client(train_loader_list, train_data_num_list, test_loader_list, device, args, model_trainer):
    trainer = FedAVGTrainer(train_loader_list, train_data_num_list, test_loader_list, device, args, model_trainer)
    client_manager = FedAVGClientManager(args, trainer)
    return client_manager
