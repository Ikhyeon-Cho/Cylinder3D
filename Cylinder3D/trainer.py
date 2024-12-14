"""
Author: Ikhyeon Cho
Link: https://github.com/Ikhyeon-Cho
File: trainer.py
Date: 2024/12/14 17:03
"""

import torch.nn as nn
from utils.train.optimizer import Optimizer
from utils.train.machine import to_device
from utils.train.evaluation import LossTracker
from utils.logger import console, tboard
import os
import torch

# Define criterion at here


class Cylinder3DTrainer:
    def __init__(self, model: nn.Module, data: dict, cfg: dict, device: str, log_dir: str):
        # Model, Data, Loss, Optimizer, Scheduler
        self.model = model.to(device)
        self.dataloader = data
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Optimizer(model, cfg).optimizer
        self.scheduler = Optimizer(model, cfg).scheduler

        # Training configs
        self.CFG = cfg
        self.device = device

        # Logger settings
        self.log_dir = log_dir
        self.logger = console.Logger(log_dir, filename='train.log')
        self.tensorboard = tboard.Logger(log_dir, ns='Train')
        self.loss_tracker = LossTracker()

        # Training variables
        self.global_step = 0
        self.best_loss = float('inf')
        self.best_epoch = 0

    def train(self):
        NUM_EPOCHS = self.CFG['epochs']
        CHECKPOINT_PERIOD = self.CFG['checkpoint_period']
        SCHEDULER_PERIOD = self.CFG['scheduler_frequency']
        SUMMARY_PERIOD = self.CFG['summary_period']

        for epoch in range(1, NUM_EPOCHS + 1):

            lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f'=> ====== Epoch [{epoch}/{NUM_EPOCHS}] ======')
            self.logger.info(f'=> Learning rate: {lr}')

            # Training setup
            self.model.train()
            self.loss_tracker.reset()

            # Batch loop
            for batch_idx, batch in enumerate(self.dataloader['train']):
                batch = to_device(batch, self.device)

                # Train
                self.optimizer.zero_grad()
                self.train_one_batch(batch)

                # Print batch loss
                batch_loss = self.loss_tracker.batch_loss()

                if (batch_idx == 0) or (batch_idx % SUMMARY_PERIOD == 0):
                    n_batches = len(self.dataloader['train'])
                    loss_print = (
                        f"=> Epoch [{epoch}/{NUM_EPOCHS}] | "
                        f"Batch: [{batch_idx+1}/{n_batches}] | "
                        f"lr: {lr:.6f} | "
                        f"Avg loss: {batch_loss['total'][-1]:.4f} "
                        f"(loss: {batch_loss['loss'][-1]:.4f})"
                    )
                    self.logger.info(loss_print)

                # Update learning rate
                if (self.scheduler is not None) and (SCHEDULER_PERIOD == 'batch'):
                    self.tensorboard.log_lr(lr, self.global_step)
                    self.scheduler.step()

                # Update global step
                self.global_step += 1

            ##################################################
            # End of batch loop -> print and log the results #
            ##################################################
            # Print epoch loss
            epoch_loss = self.loss_tracker.epoch_loss()
            self.print_epoch_loss(epoch, epoch_loss)

            # Log losses, lr to tensorboard
            self.tensorboard.log_batch_loss(
                self.loss_tracker.batch_loss(), self.loss_tracker.batch_steps())
            self.tensorboard.log_epoch_loss(epoch_loss, epoch)
            self.tensorboard.log_lr(lr, epoch)

            # Update learning rate
            if (self.scheduler is not None) and (SCHEDULER_PERIOD == 'epoch'):
                self.scheduler.step()

            # Validation
            self.validate_epoch(epoch)

            # Save checkpoint per period
            if epoch % CHECKPOINT_PERIOD == 0:
                self._save_checkpoint(
                    epoch, epoch_loss, f'LMSCNet_epoch_{epoch}')

            # Save best model if improved
            if epoch_loss["total"] < self.best_loss:
                self.best_loss = epoch_loss["total"]
                self._save_checkpoint(
                    epoch, epoch_loss, f'LMSCNet_best')

    def train_one_batch(self, batch):
        pred = self.model(batch['feats'], batch['coords'],
                          self.CFG['batch_size'])
        loss = self.criterion(pred, batch['labels_voxel'].long())

        loss.backward()
        self.optimizer.step()

        # Add current loss to loss tracker
        batch_loss_dict = {
            'loss': loss,
            'total': loss,
        }
        self.loss_tracker.append(batch_loss_dict, self.global_step)

    def validate_epoch(self, epoch):
        pass

    def print_batch_loss(self, epoch, batch_idx):
        NUM_EPOCHS = self.CFG['epochs']
        batch_loss = self.loss_tracker.batch_loss()
        lr = self.optimizer.param_groups[0]['lr']

        loss_print = "=> Epoch [{}/{}] | Batch [{}/{}] | lr: {:.6f} | Avg loss: {:.4f} (loss: {:.4f})".format(
            epoch,
            80,
            batch_idx+1,
            len(self.dataloader['train']),
            lr,
            batch_loss['total'],
            batch_loss['loss']
        )
        self.logger.info(loss_print)

    def print_epoch_loss(self, epoch):
        epoch_loss = self.loss_tracker.epoch_loss()

        self.logger.info("=> -------- Summary ---------")
        loss_print = (
            f"=>   Training Loss (Epoch {epoch}) | "
            f"Avg loss: {epoch_loss['total']:.4f} | "
            f"loss: {epoch_loss['loss']:.4f}"
        )
        self.logger.info(loss_print)

    def _save_checkpoint(self, epoch, epoch_loss, name):
        checkpoint_path = os.path.join(self.log_dir, f'{name}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': epoch_loss,
        }, checkpoint_path)
