from typing import Union
from dataclasses import dataclass

import os
from math import ceil

import torch
import torch.nn as nn
from torch.cuda import device_count
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from transformers import AdamW
from transformers.modeling_utils import PreTrainedModel


class CustomExplicitTrainer:

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module],
        args: dataclass,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        log_txt: str):

        # settings
        self.device = args.device
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.output_dir = args.output_dir
        self.num_train_epochs = args.num_train_epochs
        self.log_frequency = args.logging_steps
        self.patience = args.patience
        self.cold_steps = args.cold_steps

        # tokenizer
        self.tokenizer = None

        # loggers
        self.log_txt_file = log_txt

        # model and tokenizer
        self.model = model
        self.model.to(self.device)

        # train & eval dataloaders
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader

        # optimizer
        self.optimizer = AdamW(self.model.parameters(),
                                lr=args.cold_steps_lr)

        # scheduler (mainly for 'cold steps')
        effective_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps * device_count()
        steps_per_epoch = ceil(len(self.train_dataloader) * args.per_device_train_batch_size / effective_batch_size)
        self.lr_scheduler = MultiStepLR(self.optimizer, milestones=[self.cold_steps * steps_per_epoch], gamma=args.learning_rate / args.cold_steps_lr)


    def train(self):

        num_train_epochs = self.num_train_epochs
        steps_in_epoch = len(self.train_dataloader)
        epochs_trained = 0
        global_step = 0

        # train loop
        total_loss_scalar = 0
        best_loss, best_loss_epoch = float('inf'), 0
        for epoch in range(num_train_epochs):

            # # 'cold' steps
            # for param in self.model.parameters():
            #     param.requires_grad = False if epoch < self.cold_steps else True

            self.model.train()
            total_train_loss = 0
            for step, batch in enumerate(self.train_dataloader):

                # batch to 'device' (cuda)
                b_input_ids = batch[0].to(self.device)
                b_labels = batch[0].to(self.device)
                b_masks = batch[1].to(self.device)

                # forward
                #self.model.zero_grad()
                outputs = self.model(b_input_ids,
                                     labels=b_labels,
                                     attention_mask = b_masks,
                                     token_type_ids=None)

                loss = outputs.loss
                loss = loss / self.gradient_accumulation_steps
                batch_loss = loss.item()
                total_train_loss += batch_loss

                # backward
                loss.backward()

                # optimization step
                if step % self.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    # scheduler step & optimizer cleaning
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # book-keeping
                global_step += 1
                epochs_trained = epoch + (step + 1) / steps_in_epoch
                total_loss_scalar += loss.item()
                train_loss = total_loss_scalar / global_step

                # logging & printing (iteration level)
                if step % self.log_frequency == 0:
                    log_str = f"[train] epoch: {epochs_trained:4.2f}, loss: {train_loss:6.4f}, lr: {get_lr(self.optimizer)}"
                    print(log_str)
                    with open(self.log_txt_file, "a") as f:
                        f.write(log_str + "\n")

                del loss
                del outputs

            # eval step
            eval_loss = self.eval()

            # check for the best epoch
            if eval_loss < best_loss:
                best_loss = eval_loss
                best_loss_epoch = epoch
                self.save_model()

            self.save_model(checkpoint_name=f"epoch-checkpoint-{epochs_trained}")

            # logging & printing (epoch level)
            log_str = f"\t[eval] epoch {epochs_trained:4.2f}, loss: {eval_loss:6.4f}\n"
            print(log_str)
            with open(self.log_txt_file, "a") as f:
                f.write(log_str + "\n")            

            # early stopping
            if epoch - best_loss_epoch == self.patience:
                break


    def eval(self):

        self.model.eval()
        total_loss_scalar = 0
        global_step = 0
        for _, batch in enumerate(self.eval_dataloader):
            # batch to 'device' (cuda)
            b_input_ids = batch[0].to(self.device)
            b_labels = batch[0].to(self.device)
            b_masks = batch[1].to(self.device)

            # forward
            with torch.no_grad():
                outputs = self.model(b_input_ids,
                                        labels=b_labels,
                                        attention_mask = b_masks,
                                        token_type_ids=None)
                loss = outputs.loss

                global_step += 1
                total_loss_scalar += loss.item()
                eval_loss = total_loss_scalar / global_step

        return eval_loss


    def save_model(self, checkpoint_name="checkpoint-best"):

        output_dir = os.path.join(self.output_dir, checkpoint_name)
        state_dict = self.model.state_dict()
        self.model.save_pretrained(output_dir, state_dict=state_dict)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']