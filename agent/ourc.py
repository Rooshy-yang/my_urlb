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
    def __init__(self, update_skill_every_step, skill_dim, contrastive_scale,
                 update_encoder, contrastive_update_rate, temperature, **kwargs):
        self.skill_dim = skill_dim
        self.update_skill_every_step = update_skill_every_step
        self.contrastive_scale = contrastive_scale
        self.update_encoder = update_encoder
        self.batch_size = kwargs['batch_size']
        self.contrastive_update_rate = contrastive_update_rate
        self.temperature = temperature

        self.tau_len = update_skill_every_step
        # increase obs shape to include skill dim
        kwargs["meta_dim"] = self.skill_dim

        # create actor and critic
        super().__init__(**kwargs)

        self.tau_dim = (self.obs_dim - self.skill_dim) * self.tau_len

        # create ourc
        self.gb = GeneratorB(self.tau_dim, self.skill_dim,
                             kwargs['hidden_dim']).to(kwargs['device'])

        self.discriminator = Discriminator(self.tau_dim,
                                           self.skill_dim,
                                           kwargs['hidden_dim']).to(kwargs['device'])

        # loss criterion
        self.gb_criterion = nn.CrossEntropyLoss()

        # optimizers
        self.gb_opt = torch.optim.Adam(self.gb.parameters(), lr=self.lr)
        self.dis_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr)

        self.gb.train()
        self.discriminator.train()

        self.skill_ptr = 0
        self.skill_V = [0] * self.skill_dim
        self.skill_count = [0] * self.skill_dim
        self.skill_R = [0] * self.skill_dim
        self.ucb_scale = 2

    def get_meta_specs(self):
        return specs.Array((self.skill_dim,), np.float32, 'skill'),

    def init_meta(self):
        skill = np.zeros(self.skill_dim, dtype=np.float32)
        skill[self.skill_ptr] = 1
        meta = OrderedDict()
        meta['skill'] = skill
        return meta

    def update_meta(self, meta, global_step, time_step, finetune=False):
        if finetune:
            skill_num = meta['skill'].argmax()
            self.skill_R[skill_num] += time_step.reward
            self.skill_count[skill_num] += 1
            if global_step % self.update_skill_every_step == 0:
                skill = np.zeros(self.skill_dim, dtype=np.float32)

                v = self.skill_R[skill_num]
                # compute V(z) expectation
                self.skill_V[skill_num] = self.skill_V[skill_num] + (v - self.skill_V[skill_num]) / \
                                          self.skill_count[skill_num] * self.update_skill_every_step

                # UCB planning
                def ucb(i):
                    return self.skill_V[i] + \
                           self.ucb_scale * math.sqrt(abs(math.log(global_step + 1)) /
                                                      (self.skill_count[i] + 1e-6))

                for idx, value in enumerate(self.skill_V):
                    if ucb(idx) > ucb(self.skill_ptr):
                        self.skill_ptr = idx

                skill[self.skill_ptr] = 1
                meta = OrderedDict()
                meta['skill'] = skill

                self.skill_R = [0] * self.skill_dim
                return meta
            else:
                return meta
        else:
            if global_step % self.update_skill_every_step == 0:
                self.skill_ptr = (self.skill_ptr + 1) % self.skill_dim
                return self.init_meta()
            return meta

    def update_gb(self, skill, gb_batch, step):
        metrics = dict()
        labels = torch.argmax(skill, dim=1)
        loss, df_accuracy = self.compute_gb_loss(gb_batch, labels)

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

    def update_contrastive(self, taus, skills):
        metrics = dict()
        features = self.discriminator(taus)
        logits = self.compute_info_nce_loss(features, skills)
        loss = logits.mean()

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

    def compute_intr_reward(self, skills, tau_batch, metrics):

        # compute q(z | tau) reward
        d_pred = self.gb(tau_batch)
        d_pred_log_softmax = F.log_softmax(d_pred, dim=1)
        _, pred_z = torch.max(d_pred_log_softmax, dim=1, keepdim=True)
        gb_reward = d_pred_log_softmax[torch.arange(d_pred.shape[0]), torch.argmax(skills, dim=1)] - math.log(
            1 / self.skill_dim)
        gb_reward = gb_reward.reshape(-1, 1)

        if self.use_tb or self.use_wandb:
            metrics['gb_reward'] = gb_reward.mean().item()

        # compute contrastive reward
        features = self.discriminator(tau_batch)
        contrastive_reward = torch.exp(-self.compute_info_nce_loss(features, skills))

        intri_reward = gb_reward + contrastive_reward * self.contrastive_scale

        if self.use_tb or self.use_wandb:
            metrics['contrastive_reward'] = contrastive_reward.mean().item()

        return intri_reward

    def compute_info_nce_loss(self, features, skills):
        # label positives samples
        labels = torch.argmax(skills, dim=1)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).long()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        similarity_matrix = torch.exp(similarity_matrix / self.temperature)

        # update not only when all negative sample existed
        # assert labels[pick_one_positive_sample_idx]
        pick_one_positive_sample_idx = torch.argmax(labels, dim=-1, keepdim=True)
        pick_one_positive_sample_idx = torch.zeros_like(labels).scatter_(-1, pick_one_positive_sample_idx, 1)
        neg = (~labels.bool()).long()

        # select one and combine multiple positives
        positives = torch.sum(similarity_matrix * pick_one_positive_sample_idx, dim=-1, keepdim=True)
        negatives = torch.sum(similarity_matrix * neg, dim=-1, keepdim=True)
        eps = torch.as_tensor(1e6)
        loss = -torch.log(positives / (negatives + eps))

        return loss

    def compute_gb_loss(self, taus, skill):
        """
        DF Loss
        """

        d_pred = self.gb(taus)
        d_pred_log_softmax = F.log_softmax(d_pred, dim=1)
        _, pred_z = torch.max(d_pred_log_softmax, dim=1, keepdim=True)
        d_loss = self.gb_criterion(d_pred, skill)
        df_accuracy = torch.sum(
            torch.eq(skill,
                     pred_z.reshape(1,
                                    list(
                                        pred_z.size())[0])[0])).float() / list(
            pred_z.size())[0]
        return d_loss, df_accuracy

    def _not_allowed_update(self, skill):
        # not allowed update contrastive loss if only one positive sample in one skill
        skill_num = torch.argmax(skill, dim=-1)
        bucket = [0] * self.skill_dim
        for i in skill_num:
            bucket[i] += 1
        for num in bucket:
            if num == 1:
                return True
        return False

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        if self.reward_free:

            batch = next(replay_iter)
            tau, obs, action, reward, discount, next_obs, skill = utils.to_torch(batch, self.device)
            try_count = 0
            # while self._not_allowed_update(skill):
            #     batch = next(replay_iter)
            #     tau, obs, action, reward, discount, next_obs, skill = utils.to_torch(batch, self.device)
            #     if try_count > 3: return metrics
            #     try_count += 1
            metrics.update(self.update_contrastive(tau, skill))

            for _ in range(self.contrastive_update_rate - 1):
                # one trajectory for self.skill_dim'th tau with different skill, obs,next_obs,action from every tau,
                batch = next(replay_iter)
                tau, obs, action, reward, discount, next_obs, skill = utils.to_torch(batch, self.device)
                # while self._not_allowed_update(skill):
                #     batch = next(replay_iter)
                #     tau, obs, action, reward, discount, next_obs, skill = utils.to_torch(batch, self.device)
                #     if try_count > 3: return metrics
                #     try_count += 1

                metrics.update(self.update_contrastive(tau, skill))

            # update q(z | tau)
            # bucket count for less time spending
            metrics.update(self.update_gb(skill, tau, step))

            # compute intrinsic reward
            with torch.no_grad():
                intr_reward = self.compute_intr_reward(skill, tau, metrics)

            if self.use_tb or self.use_wandb:
                metrics['intr_reward'] = intr_reward.mean().item()

            reward = intr_reward
        else:
            batch = next(replay_iter)

            obs, action, extr_reward, discount, next_obs, skill = utils.to_torch(
                batch, self.device)
            reward = extr_reward

        # augment and encode
        obs = self.aug_and_encode(obs)
        next_obs = self.aug_and_encode(next_obs)

        if self.use_tb or self.use_wandb:
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
