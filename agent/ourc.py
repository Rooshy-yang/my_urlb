import collections
import math
from collections import OrderedDict

import hydra
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dm_env import specs

import utils
from agent.ddpg import DDPGAgent
from skill_buffer import SkillsBuffer


class GeneratorB(nn.Module):
    def __init__(self, tau_dim, skill_dim, hidden_dim):
        super().__init__()
        self.skill_pred_net = nn.Sequential(nn.Linear(tau_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, skill_dim))

        self.apply(utils.weight_init)

    def forward(self, tau):
        skill_pred = self.skill_pred_net(tau)
        return skill_pred


class Discriminator(nn.Module):
    def __init__(self, tau_dim, feature_dim, hidden_dim):
        super().__init__()
        # def SimClR :
        self.embed = nn.Sequential(nn.Linear(tau_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim),
                                   nn.ReLU(),
                                   nn.Linear(hidden_dim, feature_dim))

        self.project_head = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                          nn.ReLU(),
                                          nn.Linear(hidden_dim, feature_dim))
        self.apply(utils.weight_init)

    def forward(self, tau):
        features = self.embed(tau)
        features = self.project_head(features)
        return features


class OURCAgent(DDPGAgent):
    def __init__(self, update_skill_every_step, skill_dim, gb_scale,
                 update_encoder, contrastive_update_rate, temperature, **kwargs):
        self.skill_dim = skill_dim
        self.update_skill_every_step = update_skill_every_step
        self.gb_scale = gb_scale
        self.update_encoder = update_encoder
        self.batch_size = kwargs['batch_size']
        self.contrastive_update_rate = contrastive_update_rate
        self.temperature = temperature

        # increase obs shape to include skill dim
        kwargs["meta_dim"] = self.skill_dim

        # create actor and critic
        super().__init__(**kwargs)

        self.tau_dim = (self.obs_dim - self.skill_dim) * self.update_skill_every_step

        # create ourc
        self.gb = GeneratorB(self.tau_dim, self.skill_dim,
                             kwargs['hidden_dim']).to(kwargs['device'])

        self.discriminator = Discriminator(self.tau_dim,
                                           self.skill_dim,
                                           kwargs['hidden_dim']).to(kwargs['device'])

        # loss criterion
        self.gb_criterion = nn.CrossEntropyLoss()
        self.discriminator_criterion = nn.CrossEntropyLoss()

        # optimizers
        self.gb_opt = torch.optim.Adam(self.gb.parameters(), lr=self.lr)
        self.dis_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)

        self.gb.train()
        self.discriminator.train()

        # define a simple FIFO replay buffer for B(`| z)
        self.skillsBuffer = SkillsBuffer(self.tau_dim, self.batch_size, skill_dim)

    def add_buffer_skill2episode(self, episode, meta):
        if len(episode) == 0:
            return
        tau = np.concatenate(episode)
        skill = np.argmax(meta['skill'])
        self.skillsBuffer.store(skill, tau)

    def get_meta_specs(self):
        return specs.Array((self.skill_dim,), np.float32, 'skill'),

    def init_meta(self):
        skill = np.zeros(self.skill_dim, dtype=np.float32)
        skill[np.random.choice(self.skill_dim)] = 1.0
        meta = OrderedDict()
        meta['skill'] = skill
        return meta

    def update_meta(self, meta, global_step, time_step, finetune=False):
        if global_step % self.update_skill_every_step == 0:
            return self.init_meta()
        return meta

    def update_gb(self, skill, taus, step):
        metrics = dict()

        loss, df_accuracy = self.compute_gb_loss(taus, skill)

        self.gb_opt.zero_grad()
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.gb_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['gb_loss'] = loss.item()
            metrics['gb_acc'] = df_accuracy

        return metrics

    def update_contrastive(self, taus):
        metrics = dict()
        features = self.discriminator(taus)
        logits, labels = self.compute_info_nce_loss(features)
        loss = self.discriminator_criterion(logits, labels)

        self.dis_opt.zero_grad()
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.dis_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['contrastive_loss'] = loss.item()

        return metrics

    def compute_intr_reward(self, skill, taus4gb, step, contrastive_batch, metrics):

        # compute q(z | tau) reward
        z_hat = torch.argmax(skill, dim=1)
        d_pred = self.gb(taus4gb)
        d_pred_log_softmax = F.log_softmax(d_pred, dim=1)
        _, pred_z = torch.max(d_pred_log_softmax, dim=1, keepdim=True)
        gb_reward = d_pred_log_softmax[torch.arange(d_pred.shape[0]),
                                       z_hat] - math.log(1 / self.skill_dim)
        gb_reward = gb_reward.reshape(-1, 1)

        if self.use_tb or self.use_wandb:
            metrics['gb_reward'] = gb_reward.mean().item()

        # compute contrastive reward
        if len(contrastive_batch) < 2:
            return gb_reward
        features = self.discriminator(contrastive_batch)
        logits, labels = self.compute_info_nce_loss(features)

        logits = torch.softmax(logits, dim=1)[:, 0]
        logits = logits.view(-1, logits.shape[0] // self.skill_dim)
        logits = torch.mean(logits, dim=1)
        # assert logits.shape[0] == self.skill_dim

        skill_list = torch.argmax(skill, dim=1, keepdim=True)
        dis_reward = logits[skill_list].to(self.device)

        if self.use_tb or self.use_wandb:
            metrics.update({"Skill_{}_contrastive_reward".format(str(idx)): key.item() for idx, key in enumerate(logits)})
            metrics['dis_reward'] = dis_reward.mean().item()

        return gb_reward * self.gb_scale + dis_reward

    def compute_info_nce_loss(self, features):

        size = features.shape[0] // self.skill_dim
        labels = [[i] * size for i in range(self.skill_dim)]
        labels = torch.as_tensor(labels).view(1, -1).squeeze(0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # select one and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)[:, 0].view(labels.shape[0], -1)

        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        # set positives on column 0
        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temperature
        return logits, labels

    def compute_gb_loss(self, taus, skill):
        """
        DF Loss
        """
        z_hat = torch.argmax(skill, dim=1)
        d_pred = self.gb(taus)
        d_pred_log_softmax = F.log_softmax(d_pred, dim=1)
        _, pred_z = torch.max(d_pred_log_softmax, dim=1, keepdim=True)
        d_loss = self.gb_criterion(d_pred, z_hat)
        df_accuracy = torch.sum(
            torch.eq(z_hat,
                     pred_z.reshape(1,
                                    list(
                                        pred_z.size())[0])[0])).float() / list(
            pred_z.size())[0]
        return d_loss, df_accuracy

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)

        obs, action, extr_reward, discount, next_obs, skill = utils.to_torch(
            batch, self.device)

        # augment and encode
        obs = self.aug_and_encode(obs)
        next_obs = self.aug_and_encode(next_obs)

        if self.reward_free:
            # update q(z | tau)

            # bucket count for less time spending
            skill_tmp = skill.clone().detach()
            skill_num = torch.argmax(skill_tmp.cpu(), 1).numpy()
            counts = [0] * self.skill_dim
            for i in skill_num:
                counts[i] += 1
            generator_b_batch = self.skillsBuffer.sample_batch(batch_size=self.batch_size, counts=counts)
            generator_b_batch = generator_b_batch.to(self.device)
            metrics.update(self.update_gb(skill_tmp, generator_b_batch, step))

            # update contrastive, tau_batch_size not larger than self.batch_size
            tau_batch_size = min(min(self.skillsBuffer.sizes), self.batch_size // self.skill_dim)
            contrastive_batch = torch.as_tensor([])

            if tau_batch_size == 1:
                if self.use_tb or self.use_wandb:
                    metrics['contrastive_loss'] = 0
                    metrics.update({"Skill_{}_contrastive_reward".format(str(idx)): key for idx, key in enumerate([0] * self.skill_dim)})
                    metrics['dis_reward'] = 0
            else:
                for _ in range(self.contrastive_update_rate):
                    contrastive_batch = self.skillsBuffer.sample_batch(batch_size=tau_batch_size * self.skill_dim,
                                                                       counts=[tau_batch_size] * self.skill_dim)
                    contrastive_batch = contrastive_batch.to(self.device)
                    metrics.update(self.update_contrastive(contrastive_batch))

            # compute intrinsic reward
            with torch.no_grad():
                intr_reward = self.compute_intr_reward(skill, generator_b_batch, step, contrastive_batch, metrics)

            if self.use_tb or self.use_wandb:
                metrics['intr_reward'] = intr_reward.mean().item()
                metrics['tau_batch_size'] = tau_batch_size
            reward = intr_reward
        else:
            reward = extr_reward

        if self.use_tb or self.use_wandb:
            metrics['extr_reward'] = extr_reward.mean().item()
            metrics['batch_reward'] = reward.mean().item()

        if not self.update_encoder:
            obs = obs.detach()
            next_obs = next_obs.detach()

        # extend observations with skill
        obs = torch.cat([obs, skill], dim=1)
        next_obs = torch.cat([next_obs, skill], dim=1)

        # update critic
        metrics.update(
            self.update_critic(obs.detach(), action, reward, discount,
                               next_obs.detach(), step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
