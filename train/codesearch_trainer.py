import logging
import math
import os
import time
from collections import OrderedDict

import torch
from tqdm import tqdm, trange
import numpy as np
from transformers import get_linear_schedule_with_warmup, AdamW


class CodeSearchTrainer:
    def __init__(self, args, device, model, train_dl=None, valid_dl=None, test_dl=None):
        self.args = args
        self.device = device

        # self.num_labels = args.num_labels
        self.set_data(train_dl, valid_dl, test_dl)

        self.model = model
        self.global_model_params = model.state_dict()

        self.results = {}
        self.best_accuracy = 0.0

        self.freeze_layers = args.freeze_layers.split(",") if args.freeze_layers else []

    def set_data(self, train_dl=None, valid_dl=None, test_dl=None):
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.test_dl = test_dl

    def get_model_params(self):
        params = OrderedDict()
        for key, value in self.model.state_dict().items():
            params[key] = value.clone().detach()
        return params

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def set_global_model_params(self, params):
        self.global_model_params = params

    def get_global_model_params(self):
        return self.global_model_params

    def train(self):
        """ Train the model """
        # if args.local_rank in [-1, 0]:
        #     tb_writer = SummaryWriter()
        #
        # args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
        # train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        # train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
        #
        # if args.max_steps > 0:
        #     t_total = args.max_steps
        #     args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
        # else:
        #     t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        #
        # scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, t_total)
        iteration_in_total = len(self.train_dl) // self.args.gradient_accumulation_steps * self.args.epochs
        optimizer, scheduler = self.build_optimizer(self.model, iteration_in_total)

        # checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
        # scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
        # if os.path.exists(scheduler_last):
        #     scheduler.load_state_dict(torch.load(scheduler_last))

        # Train!
        logging.info("***** Running training *****")
        # logging.info("  Num examples = %d", len(train_dataset))
        # logging.info("  Num Epochs = %d", args.num_train_epochs)
        # logging.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
        # logging.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
        #              args.train_batch_size * args.gradient_accumulation_steps * (
        #                  torch.distributed.get_world_size() if args.local_rank != -1 else 1))
        # logging.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        # logging.info("  Total optimization steps = %d", t_total)

        # global_step = args.start_step
        args = self.args
        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        best_acc = 0.0
        self.model.zero_grad()
        # train_iterator = trange(args.start_epoch, int(args.num_train_epochs), desc="Epoch",
        #                         disable=args.local_rank not in [-1, 0])
        # set_seed(args)  # Added here for reproductibility (even between python2 2 and 3)
        self.model.train()
        # for idx, _ in enumerate(train_iterator):
        for idx in range(args.epochs):
            tr_loss = 0.0
            for step, batch in enumerate(self.train_dl):

                if step % 500 == 0:
                    logging.info("step: %d,time:%s", step, time.asctime(time.localtime(time.time())))

                batch = tuple(t.to(self.device) for t in batch)
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
                          # XLM don't use segment_ids
                          'labels': batch[3]}
                ouputs = self.model(**inputs)
                loss = ouputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

                # if args.n_gpu > 1:
                #     loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                # if args.fp16:
                #     try:
                #         from apex import amp
                #     except ImportError:
                #         raise ImportError(
                #             "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
                #     with amp.scale_loss(loss, optimizer) as scaled_loss:
                #         scaled_loss.backward()
                #     torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                # else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                if step % 100 == 0:
                    logging.info("epoch = %d, batch_idx = %d/%d, loss = %s" % (
                        idx, step, len(self.train_dl) - 1, loss))

                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    # if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    #     # Log metrics
                    #     if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                    #         results = evaluate(args, model, tokenizer, checkpoint=str(global_step))
                    #         for key, value in results.items():
                    #             tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    #             logger.info('loss %s', str(tr_loss - logging_loss))
                    #     tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    #     tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    #     logging_loss = tr_loss
                # if args.max_steps > 0 and global_step > args.max_steps:
                # epoch_iterator.close()
                # break

            # if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
            #     results = self.evaluate(args, self.model, tokenizer, checkpoint=str(args.start_epoch + idx))
            #
            #     last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
            #     if not os.path.exists(last_output_dir):
            #         os.makedirs(last_output_dir)
            #     model_to_save = model.module if hasattr(model,
            #                                             'module') else model  # Take care of distributed/parallel training
            #     model_to_save.save_pretrained(last_output_dir)
            #     logging.info("Saving model checkpoint to %s", last_output_dir)
            #     idx_file = os.path.join(last_output_dir, 'idx_file.txt')
            #     with open(idx_file, 'w', encoding='utf-8') as idxf:
            #         idxf.write(str(args.start_epoch + idx) + '\n')
            #
            #     torch.save(optimizer.state_dict(), os.path.join(last_output_dir, "optimizer.pt"))
            #     torch.save(scheduler.state_dict(), os.path.join(last_output_dir, "scheduler.pt"))
            #     logging.info("Saving optimizer and scheduler states to %s", last_output_dir)
            #
            #     step_file = os.path.join(last_output_dir, 'step_file.txt')
            #     with open(step_file, 'w', encoding='utf-8') as stepf:
            #         stepf.write(str(global_step) + '\n')
            #
            #     if (results['acc'] > best_acc):
            #         best_acc = results['acc']
            #         output_dir = os.path.join(args.output_dir, 'checkpoint-best')
            #         if not os.path.exists(output_dir):
            #             os.makedirs(output_dir)
            #         model_to_save = model.module if hasattr(model,
            #                                                 'module') else model  # Take care of distributed/parallel training
            #         model_to_save.save_pretrained(output_dir)
            #         torch.save(args, os.path.join(output_dir, 'training_{}.bin'.format(idx)))
            #         logging.info("Saving model checkpoint to %s", output_dir)
            #
            #         torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            #         torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            #         logging.info("Saving optimizer and scheduler states to %s", output_dir)

            # if args.max_steps > 0 and global_step > args.max_steps:
            #     train_iterator.close()
            #     break

        # if args.local_rank in [-1, 0]:
        #     tb_writer.close()

        return global_step, tr_loss / global_step

    # def eval(self):
    #     # Loop to handle MNLI double evaluation (matched, mis-matched)
    #     eval_task_names = (args.task_name,)
    #     eval_outputs_dirs = (args.output_dir,)
    #
    #     results = {}
    #     for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
    #         if (mode == 'dev'):
    #             eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, ttype='dev')
    #         elif (mode == 'test'):
    #             eval_dataset, instances = load_and_cache_examples(args, eval_task, tokenizer, ttype='test')
    #
    #         if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
    #             os.makedirs(eval_output_dir)
    #
    #         args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    #         # Note that DistributedSampler samples randomly
    #         eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(
    #             eval_dataset)
    #         eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    #
    #         # Eval!
    #         logging.info("***** Running evaluation {} *****".format(prefix))
    #         logging.info("  Num examples = %d", len(eval_dataset))
    #         logging.info("  Batch size = %d", args.eval_batch_size)
    #         eval_loss = 0.0
    #         nb_eval_steps = 0
    #         preds = None
    #         out_label_ids = None
    #         for batch in tqdm(eval_dataloader, desc="Evaluating"):
    #             self.model.eval()
    #             batch = tuple(t.to(args.device) for t in batch)
    #
    #             with torch.no_grad():
    #                 inputs = {'input_ids': batch[0],
    #                           'attention_mask': batch[1],
    #                           'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,
    #                           # XLM don't use segment_ids
    #                           'labels': batch[3]}
    #
    #                 outputs = model(**inputs)
    #                 tmp_eval_loss, logits = outputs[:2]
    #
    #                 eval_loss += tmp_eval_loss.mean().item()
    #             nb_eval_steps += 1
    #             if preds is None:
    #                 preds = logits.detach().cpu().numpy()
    #                 out_label_ids = inputs['labels'].detach().cpu().numpy()
    #             else:
    #
    #                 preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
    #
    #                 out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
    #         # eval_accuracy = accuracy(preds,out_label_ids)
    #         eval_loss = eval_loss / nb_eval_steps
    #         if args.output_mode == "classification":
    #             preds_label = np.argmax(preds, axis=1)
    #         result = compute_metrics(eval_task, preds_label, out_label_ids)
    #         results.update(result)
    #         if (mode == 'dev'):
    #             output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    #             with open(output_eval_file, "a+") as writer:
    #                 logging.info("***** Eval results {} *****".format(prefix))
    #                 writer.write('evaluate %s\n' % checkpoint)
    #                 for key in sorted(result.keys()):
    #                     logging.info("  %s = %s", key, str(result[key]))
    #                     writer.write("%s = %s\n" % (key, str(result[key])))
    #         elif (mode == 'test'):
    #             output_test_file = args.test_result_dir
    #             output_dir = os.path.dirname(output_test_file)
    #             if not os.path.exists(output_dir):
    #                 os.makedirs(output_dir)
    #             with open(output_test_file, "w") as writer:
    #                 logging.info("***** Output test results *****")
    #                 all_logits = preds.tolist()
    #                 for i, logit in tqdm(enumerate(all_logits), desc='Testing'):
    #                     instance_rep = '<CODESPLIT>'.join(
    #                         [item.encode('ascii', 'ignore').decode('ascii') for item in instances[i]])
    #
    #                     writer.write(instance_rep + '<CODESPLIT>' + '<CODESPLIT>'.join([str(l) for l in logit]) + '\n')
    #                 for key in sorted(result.keys()):
    #                     print("%s = %s" % (key, str(result[key])))
    #
    #     return results

    def build_optimizer(self, model, iteration_in_total):
        warmup_steps = math.ceil(iteration_in_total * self.args.warmup_ratio)
        # self.args.warmup_steps = warmup_steps if self.args.warmup_steps == 0 else self.args.warmup_steps
        logging.info("warmup steps = %d" % warmup_steps)
        self.freeze_model_parameters(model)
        optimizer = AdamW(model.parameters(), lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        # logging.info('warmup steps:%d' % warmup_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=iteration_in_total
        )
        return optimizer, scheduler

    def freeze_model_parameters(self, model):
        modules = list()
        logging.info("freeze layers: %s" % str(self.freeze_layers))
        for layer_idx in self.freeze_layers:
            if layer_idx == "e":
                modules.append(model.distilbert.embeddings)
            else:
                modules.append(model.distilbert.transformer.layer[int(layer_idx)])
        for module in modules:
            for param in module.parameters():
                param.requires_grad = False
        logging.info(self.get_parameter_number(model))

    def get_parameter_number(self, net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}
