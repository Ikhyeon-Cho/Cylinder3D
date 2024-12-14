"""
Author: Ikhyeon Cho
Link: https://github.com/Ikhyeon-Cho
File: trainer.py
Date: 2024/12/14 17:03
"""

import torch
import torch.nn as nn
from utils.train.optimizer import Optimizer
from utils.train.machine import to_device
from utils.train.evaluation import LossTracker
from utils.train import checkpoint
from utils.logger import console, tboard


class Cylinder3DTrainer:
    def __init__(self, model: nn.Module, data: dict, cfg: dict, device: str, log_dir: str):
        # Model, Data, Loss, Optimizer, Scheduler
        self.model = model.to(device)
        self.dataloader = data
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
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
        self.best_val_loss = float('inf')
        self.best_epoch = 0

        # Load checkpoint
        print('=> Loading checkpoint...')
        epoch, loss, optimizer = checkpoint.load(self.model,
                                                 self.CFG['checkpoint_path'],
                                                 device=self.device,
                                                 strict=True)
        if epoch > 0:
            self.best_epoch = epoch
            self.best_val_loss = loss
            self.optimizer = optimizer
            self.logger.info(f'=> Loaded checkpoint from epoch {epoch}')

    def train(self):
        NUM_EPOCHS = self.CFG['epochs']
        CHECKPOINT_PERIOD = self.CFG['checkpoint_period']
        SCHEDULER_PERIOD = self.CFG['scheduler_frequency']
        SUMMARY_PERIOD = self.CFG['summary_period']

        print(f'=> Start training...')

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
                if (batch_idx == 0) or (batch_idx % SUMMARY_PERIOD == 0):
                    self.print_batch_loss(epoch, batch_idx)

                # Update learning rate
                if (self.scheduler is not None) and (SCHEDULER_PERIOD == 'batch'):
                    self.tensorboard.log_lr(lr, self.global_step)
                    self.scheduler.step()

                # Update global step
                self.global_step += 1

            ##################################################
            # End of batch loop -> print and log the results #
            ##################################################
            self.logger.info("=> -------- Summary ---------")
            epoch_loss = self.loss_tracker.epoch_loss()
            self.print_epoch_loss(epoch, epoch_loss, phase='train')

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
                checkpoint.save(self.model, self.optimizer, epoch, epoch_loss,
                                self.log_dir + f'Cylinder3D_epoch_{epoch}.pth')

    def train_one_batch(self, batch: dict):

        pred = self.model(batch['feats'], batch['coords'],
                          self.CFG['batch_size'])
        loss = self.criterion(pred, batch['labels_voxel'])
        loss.backward()
        self.optimizer.step()

        # Add current loss to loss tracker
        batch_loss_dict = {
            'loss': loss,
            'total': loss,
        }
        self.loss_tracker.append(batch_loss_dict, self.global_step)

    def print_batch_loss(self, epoch: int, batch_idx: int):

        NUM_EPOCHS = self.CFG['epochs']
        batch_loss = self.loss_tracker.batch_loss()
        lr = self.optimizer.param_groups[0]['lr']
        n_batches = len(self.dataloader['train'])

        loss_print = (
            f"=> Epoch [{epoch}/{NUM_EPOCHS}] | "
            f"Batch: [{batch_idx+1}/{n_batches}] | "
            f"lr: {lr:.6f} | "
            f"Avg loss: {batch_loss['total'][-1]:.4f} "
            f"(loss: {batch_loss['loss'][-1]:.4f})"
        )
        self.logger.info(loss_print)

    def print_epoch_loss(self, epoch: int, epoch_loss: dict, phase: str):

        loss_print = (
            f"=>   {phase} loss (Epoch {epoch}) | "
            f"Avg loss: {epoch_loss['total']:.4f} | "
            f"loss: {epoch_loss['loss']:.4f}"
        )
        self.logger.info(loss_print)

    def validate_epoch(self, epoch: int):

        VALIDATION_SUMMARY_PERIOD = self.CFG['validation_summary_period']
        val_loss_tracker = LossTracker()
        val_tensorboard = tboard.Logger(self.log_dir, ns='Val')

        self.model.eval()
        with torch.no_grad():
            for batch in self.dataloader['val']:
                batch = to_device(batch, self.device)

                pred = self.model(batch['feats'], batch['coords'],
                                  self.CFG['batch_size'])
                loss = self.criterion(pred, batch['labels_voxel'])
                loss_dict = {
                    'loss': loss,
                    'total': loss,
                }
                val_loss_tracker.append(loss_dict)

            # End of batch loop -> print and log the results
            val_epoch_loss = val_loss_tracker.epoch_loss()

            # Log val loss to tensorboard
            val_tensorboard.log_epoch_loss(
                val_loss_tracker.epoch_loss(), epoch)

            # Print val loss
            if (epoch == 0) or (epoch % VALIDATION_SUMMARY_PERIOD == 0):
                self.print_epoch_loss(epoch, val_epoch_loss, phase='val')
                self.logger.info("=> --------------------------")

            # Save best model if improved
            if val_epoch_loss["total"] < self.best_val_loss:
                self.best_val_loss = val_epoch_loss["total"]
                checkpoint.save(self.model, self.optimizer, epoch, val_epoch_loss,
                                self.log_dir + f'Cylinder3D_best.pth')
