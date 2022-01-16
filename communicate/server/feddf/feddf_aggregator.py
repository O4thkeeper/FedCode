import logging

import torch
from tqdm import tqdm

from communicate.server.base.base_aggregator import BaseAggregator
from utils.model_utils import copy_state_dict
import torch.nn.functional as F


class FedDfAggregator(BaseAggregator):

    def __init__(self, server_trainer, client_trainer, data_loader, args, device):
        super().__init__(client_trainer)
        self.args = args
        self.worker_num = args.client_num_in_total
        self.device = device
        self.server_trainer = server_trainer
        self.client_trainer = client_trainer
        self.data_loader = data_loader

    def aggregate(self, model_params_list, sample_num_list):
        averaged_params = self.avg_params(model_params_list, sample_num_list)

        self.knowledge_transfer(self.server_trainer, averaged_params, self.client_trainer, model_params_list,
                                self.data_loader, self.args)
        self.set_global_model_params(self.server_trainer.get_global_model_params())
        # filename = os.path.join('cache', str(time.time()))
        # torch.save(averaged_params, filename)

    def test_on_server(self):
        self.trainer.test()

    def avg_params(self, model_params_list, sample_num_list):
        training_num = sum(sample_num_list)

        averaged_params = copy_state_dict(model_params_list[0])
        for i in range(len(model_params_list)):
            local_sample_number = sample_num_list[i]
            local_model_params = model_params_list[i]
            w = local_sample_number / training_num

            logging.info('average model of client %d with w %s' % (i, w))

            for k in averaged_params.keys():
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params

    def knowledge_transfer(self, server_trainer, avg_params, client_trainer, client_params_list, data_loader, args):
        server_model = server_trainer.get_model()
        optimizer_server = torch.optim.Adam(server_model.parameters(), lr=args.server_lr)

        for step, batch in tqdm(enumerate(data_loader)):
            batch = tuple(t.to(self.device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                      'labels': batch[3]}
            teacher_logits = []
            client_trainer.get_model().to(self.device)
            for client_params in client_params_list:
                client_trainer.set_model_params(client_params)
                with torch.no_grad():
                    teacher_logits.append(client_trainer.model(**inputs)[1])
            client_trainer.get_model().cpu()
            weights = [1.0 / len(client_params_list)] * len(client_params_list)
            teacher_avg_logits = sum([teacher_logit * weight for teacher_logit, weight in zip(teacher_logits, weights)])

            server_trainer.set_model_params(avg_params)
            server_model.to(self.device)
            server_model.train()
            for i in range(args.server_local_steps):
                student_logits = server_model(**inputs)[1]
                student_avg_loss = self.divergence(student_logits, teacher_avg_logits)
                optimizer_server.zero_grad()
                student_avg_loss.backward()
                torch.nn.utils.clip_grad_norm_(server_model.parameters(), 5)
                optimizer_server.step()
            server_model.cpu()
        server_trainer.set_global_model_params(server_trainer.get_model_params())

    def divergence(self, student_logits, teacher_logits, use_teacher_logits=True):
        divergence = F.kl_div(
            F.log_softmax(student_logits, dim=1),
            F.softmax(teacher_logits, dim=1)
            if use_teacher_logits
            else teacher_logits,
            reduction="batchmean",
        )
        return divergence
