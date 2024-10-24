# -*- coding: utf-8 -*-
import os, sys

import torch
import torch.nn as nn
from torch.optim import Adam
import lightning as L
import pysnooper


class DynamicRegressionModel(nn.Module):
    def __init__(self, input_dim, dropout_rate=0.1, output_dim=1):
        super().__init__()
        self.input_dim = input_dim
        self.dropout_rate = dropout_rate

        self.activation = nn.LeakyReLU()
        # self.out_activation = nn.Softplus()

        # first layer
        self.fc1 = nn.Linear(input_dim, 2 * input_dim)
        self.bn1 = nn.BatchNorm1d(2 * input_dim)
        self.dropout1 = nn.Dropout(self.dropout_rate)

        # second layer
        self.fc2 = nn.Linear(2 * input_dim, 4 * input_dim)
        self.bn2 = nn.BatchNorm1d(4 * input_dim)
        self.dropout2 = nn.Dropout(self.dropout_rate)

        # third layer
        self.fc3 = nn.Linear(4 * input_dim, 8 * input_dim)
        self.bn3 = nn.BatchNorm1d(8 * input_dim)
        self.dropout3 = nn.Dropout(self.dropout_rate)

        # fourth layer
        self.fc4 = nn.Linear(8 * input_dim, 4 * input_dim)
        self.bn4 = nn.BatchNorm1d(4 * input_dim)
        self.dropout4 = nn.Dropout(self.dropout_rate)

        # fifth layer
        self.fc5 = nn.Linear(4 * input_dim, 2 * input_dim)
        self.bn5 = nn.BatchNorm1d(2 * input_dim)
        self.dropout5 = nn.Dropout(self.dropout_rate)

        # sixth layer
        self.fc6 = nn.Linear(2 * input_dim, input_dim)
        self.bn6 = nn.BatchNorm1d(input_dim)
        self.dropout6 = nn.Dropout(self.dropout_rate)

        # output layer
        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x1 = self.activation(self.bn1(self.fc1(x)))
        if self.input_dim > 2:
            x1 = self.dropout1(x1)

        x2 = self.activation(self.bn2(self.fc2(x1)))
        x2 = self.dropout2(x2)

        x3 = self.activation(self.bn3(self.fc3(x2)))
        x3 = self.dropout3(x3)

        x4 = self.activation(self.bn4(self.fc4(x3)))
        x4 = self.dropout4(x4)
        x4 += x2

        x5 = self.activation(self.bn5(self.fc5(x4)))
        x5 = self.dropout5(x5)
        x5 += x1

        x6 = self.activation(self.bn6(self.fc6(x5)))
        if self.input_dim > 2:
            x6 = self.dropout6(x6)
        x6 += x

        x7 = self.fc_out(x6)
        # x7 = self.out_activation(x7)
        return x7


class MultiModelContainer(L.LightningModule):
    def __init__(self, models, compounds_reactions_np, reaction_names):
        super().__init__()
        self.automatic_optimization = False  # Disable automatic optimization

        self.models = nn.ModuleDict(
            {reaction_name: model for reaction_name, model in models.items()}
        )
        self.compounds_reactions_tensor = compounds_reactions_np
        self.reaction_names = reaction_names

        self.predictions = []

    def setup(self, stage=None):
        self.compounds_reactions_tensor = torch.tensor(
            self.compounds_reactions_tensor, dtype=torch.float32, device=self.device
        )

    def forward(self, x, reaction_name):
        outputs = self.models[reaction_name](x)
        return outputs

    def training_step(self, batch, batch_idx):
        optimizers = self.optimizers()

        samples_reactions_scfea = []
        samples_reactions_geneMean = []
        cur_batch_size = batch[list(batch.keys())[0]].shape[0]
        for reaction_name in self.reaction_names:
            if self.models[reaction_name] is None:
                output = torch.zeros(
                    cur_batch_size, 1, device=self.device, dtype=torch.float32
                )
                samples_reactions_scfea.append(output)
                samples_reactions_geneMean.append(
                    torch.ones(cur_batch_size, device=self.device, dtype=torch.float32)
                )
                continue

            x = None
            x = batch[reaction_name]
            output = self.forward(x, reaction_name)
            samples_reactions_scfea.append(output)
            samples_reactions_geneMean.append(x.mean(dim=1))

        samples_reactions_scfea = (
            torch.stack(samples_reactions_scfea).transpose(0, 1).squeeze(2)
        )

        samples_reactions_geneMean = torch.stack(samples_reactions_geneMean).transpose(
            0, 1
        )
        (total_loss, reaction_cor_loss, sample_cor_loss, imbalance_loss, cv_loss) = (
            self.get_total_loss(samples_reactions_scfea, samples_reactions_geneMean)
        )

        self.log(
            "train_total_loss",
            total_loss.to(self.device),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train_imbalance_loss",
            imbalance_loss.to(self.device),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train_cv_loss",
            cv_loss.to(self.device),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train_reaction_cor_loss",
            reaction_cor_loss.to(self.device),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train_sample_cor_loss",
            sample_cor_loss.to(self.device),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        # Directly after the manual_backward call in your training_step

        self.manual_backward(total_loss)
        for optimizer in optimizers:
            optimizer.step()
            optimizer.zero_grad()

        return total_loss

    def validation_step(self, batch, batch_idx):
        samples_reactions_scfea = []
        samples_reactions_geneMean = []
        cur_batch_size = batch[list(batch.keys())[0]].shape[0]
        for reaction_name in self.reaction_names:
            if self.models[reaction_name] is None:
                output = torch.zeros(
                    cur_batch_size, 1, device=self.device, dtype=torch.float32
                )
                samples_reactions_scfea.append(output)
                samples_reactions_geneMean.append(
                    torch.ones(cur_batch_size, device=self.device, dtype=torch.float32)
                )
                continue

            x = None
            x = batch[reaction_name]
            output = self.forward(x, reaction_name)
            samples_reactions_scfea.append(output)
            samples_reactions_geneMean.append(x.mean(dim=1))

        samples_reactions_scfea = (
            torch.stack(samples_reactions_scfea).transpose(0, 1).squeeze(2)
        )

        samples_reactions_geneMean = torch.stack(samples_reactions_geneMean).transpose(
            0, 1
        )
        (total_loss, reaction_cor_loss, sample_cor_loss, imbalance_loss, cv_loss) = (
            self.get_total_loss(samples_reactions_scfea, samples_reactions_geneMean)
        )
        self.log(
            "val_total_loss",
            total_loss.to(self.device),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val_imbalance_loss",
            imbalance_loss.to(self.device),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val_cv_loss",
            cv_loss.to(self.device),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val_reaction_cor_loss",
            reaction_cor_loss.to(self.device),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "val_sample_cor_loss",
            sample_cor_loss.to(self.device),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        return total_loss

    def test_step(self, batch, batch_idx):
        samples_reactions_scfea = []
        samples_reactions_geneMean = []
        cur_batch_size = batch[list(batch.keys())[0]].shape[0]
        for reaction_name in self.reaction_names:
            if self.models[reaction_name] is None:
                output = torch.zeros(
                    cur_batch_size, 1, device=self.device, dtype=torch.float32
                )
                samples_reactions_scfea.append(output)
                samples_reactions_geneMean.append(
                    torch.ones(cur_batch_size, device=self.device, dtype=torch.float32)
                )
                continue

            x = None
            x = batch[reaction_name]
            output = self.forward(x, reaction_name)
            samples_reactions_scfea.append(output)
            samples_reactions_geneMean.append(x.mean(dim=1))

        samples_reactions_scfea = (
            torch.stack(samples_reactions_scfea).transpose(0, 1).squeeze(2)
        )

        self.predictions.append(samples_reactions_scfea.detach())

        samples_reactions_geneMean = torch.stack(samples_reactions_geneMean).transpose(
            0, 1
        )
        (total_loss, reaction_cor_loss, sample_cor_loss, imbalance_loss, cv_loss) = (
            self.get_total_loss(samples_reactions_scfea, samples_reactions_geneMean)
        )
        self.log("test_total_loss", total_loss.to(self.device))
        self.log("test_imbalance_loss", imbalance_loss.to(self.device))
        self.log("test_cv_loss", cv_loss.to(self.device))
        self.log("test_reaction_cor_loss", reaction_cor_loss.to(self.device))
        self.log("test_sample_cor_loss", sample_cor_loss.to(self.device))

        return total_loss

    def on_test_epoch_end(self):
        self.predictions = torch.cat(self.predictions, dim=0).detach().cpu().numpy()
        self.predictions = abs(self.predictions)

    def get_total_loss(self, samples_reactions, samples_reactions_geneMean):

        # coefficient of variation loss
        #'''
        cv_loss = samples_reactions.std(dim=0) / (samples_reactions.mean(dim=0) + 1e-8)
        # fille na with 0
        cv_loss[torch.isnan(cv_loss)] = 0
        cv_loss /= cv_loss.sum()
        cv_loss = 1 - cv_loss
        cv_loss = cv_loss.mean()
        #'''

        # pearson correlation loss by column
        reaction_cor_loss = torch.zeros(samples_reactions.shape[1])
        for i in range(samples_reactions.shape[1]):
            x = samples_reactions[:, i]
            y = samples_reactions_geneMean[:, i]

            mean_x = torch.mean(x)
            mean_y = torch.mean(y)

            x_centered = x - mean_x
            y_centered = y - mean_y

            cov_xy = torch.mean(x_centered * y_centered)

            std_x = torch.std(x_centered, unbiased=False)
            std_y = torch.std(y_centered, unbiased=False)

            pearson_corr = cov_xy / (std_x * std_y)
            if torch.isnan(pearson_corr) or torch.isinf(pearson_corr):
                pearson_corr = 0

            reaction_cor_loss[i] = 1 - pearson_corr
        # fill na with 0
        reaction_cor_loss[torch.isnan(reaction_cor_loss)] = 0
        reaction_cor_loss = reaction_cor_loss.mean()

        # pearson correlation loss by row
        sample_cor_loss = torch.zeros(samples_reactions.shape[0])
        for i in range(samples_reactions.shape[0]):
            x = samples_reactions[i]
            y = samples_reactions_geneMean[i]

            mean_x = torch.mean(x)
            mean_y = torch.mean(y)

            x_centered = x - mean_x
            y_centered = y - mean_y

            cov_xy = torch.mean(x_centered * y_centered)

            std_x = torch.std(x_centered, unbiased=False)
            std_y = torch.std(y_centered, unbiased=False)

            pearson_corr = cov_xy / (std_x * std_y)
            if torch.isnan(pearson_corr) or torch.isinf(pearson_corr):
                pearson_corr = 0

            sample_cor_loss[i] = 1 - pearson_corr
        # fill na with 0
        sample_cor_loss[torch.isnan(sample_cor_loss)] = 0
        sample_cor_loss = sample_cor_loss.mean()

        # imbalance loss
        row_sum = samples_reactions.sum(dim=1) + 1e-8
        # normalize each row by row sum
        samples_reactions = samples_reactions / row_sum.view(-1, 1)
        imbalance_loss_list = []
        for i in range(samples_reactions.shape[0]):
            cur_imbalance_loss = samples_reactions[i] * self.compounds_reactions_tensor
            cur_imbalance_loss = cur_imbalance_loss.sum(dim=1).pow(2).mean()
            imbalance_loss_list.append(cur_imbalance_loss)
        imbalance_loss_list = torch.stack(imbalance_loss_list)
        # fill na with 0
        imbalance_loss_list[torch.isnan(imbalance_loss_list)] = 0
        imbalance_loss = imbalance_loss_list.mean()

        # weights
        lambda_imbalance_loss = 0.5
        lambda_cv = 0.005
        lambda_reaction_cor_loss = 0.2
        lambda_sample_cor_loss = 0.195

        # if they are nan or inf, set them to 0
        #'''
        if torch.isnan(imbalance_loss):
            imbalance_loss = torch.zeros(1)
        if torch.isnan(cv_loss):
            cv_loss = torch.zeros(1)
        if torch.isnan(reaction_cor_loss):
            reaction_cor_loss = torch.zeros(1)
        if torch.isnan(sample_cor_loss):
            sample_cor_loss = torch.zeros(1)
        #'''

        total_loss = (
            lambda_imbalance_loss * imbalance_loss
            + lambda_reaction_cor_loss * reaction_cor_loss
            + lambda_sample_cor_loss * sample_cor_loss
            + lambda_cv * cv_loss
        )
        return (total_loss, reaction_cor_loss, sample_cor_loss, imbalance_loss, cv_loss)

    def configure_optimizers(self):
        optimizers = [
            Adam(self.models[reaction_name].parameters(), lr=0.01, weight_decay=1e-3)
            for reaction_name in self.reaction_names
            if self.models[reaction_name] is not None
        ]
        return optimizers
