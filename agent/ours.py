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


class OURS(nn.Module):
    def __init__(self, tau_dim, skill_dim, hidden_dim):
        super().__init__()
        self.skill_pred_net = nn.Sequential(nn.Linear(tau_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, hidden_dim),
                                            nn.ReLU(),
                                            nn.Linear(hidden_dim, skill_dim))

        self.apply(utils.weight_init)

    #Generator B
    def forward(self, tau):
        skill_pred = self.skill_pred_net(tau)
        return skill_pred


class OURSAgent(DDPGAgent):
    def __init__(self, update_skill_every_step, skill_dim, ours_scale,
                 update_encoder, **kwargs):
        self.skill_dim = skill_dim
        self.update_skill_every_step = update_skill_every_step
        self.ours_scale = ours_scale
        self.update_encoder = update_encoder
        # increase obs shape to include skill dim
        kwargs["meta_dim"] = self.skill_dim

        # create actor and critic
        super().__init__(**kwargs)

        # create ours
        self.ours = OURS((self.obs_dim - self.skill_dim) * self.update_skill_every_step, self.skill_dim,
                           kwargs['hidden_dim']).to(kwargs['device'])

        # loss criterion
        self.ours_criterion = nn.CrossEntropyLoss()
        # optimizers
        self.ours_opt = torch.optim.Adam(self.ours.parameters(), lr=self.lr)

        self.ours.train()

        # define B(`| z)
        self.skills2taus = {i: collections.deque(maxlen=kwargs['replay_buffer_size'] // skill_dim)
                            for i in range(skill_dim)}

    def add_buffer_skill2episode(self, episode, meta):
        # assert len(episode) == 1 and episode.keys == meta['skill']
        for skill in episode.keys():
            if len(episode[skill]) != self.update_skill_every_step:
                continue
            tau = np.concatenate(episode[skill])
            self.skills2taus[skill].append(tau)

    def get_meta_specs(self):
        return (specs.Array((self.skill_dim,), np.float32, 'skill'),)

    def init_meta(self):
        skill = np.zeros(self.skill_dim, dtype=np.float32)
        skill[np.random.choice(self.skill_dim)] = 1.0
        meta = OrderedDict()
        meta['skill'] = skill
        return meta

    def update_meta(self, meta, global_step, time_step):
        if global_step % self.update_skill_every_step == 0:
            return self.init_meta()
        return meta

    def update_ours(self, skill, trajectory, step):
        metrics = dict()

        loss, df_accuracy = self.compute_ours_loss(trajectory, skill)

        self.ours_opt.zero_grad()
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.ours_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['ours_loss'] = loss.item()
            metrics['ours_acc'] = df_accuracy

        return metrics

    def compute_intr_reward(self, skill, tau, step):
        z_hat = torch.argmax(skill, dim=1)
        d_pred = self.ours(tau)
        d_pred_log_softmax = F.log_softmax(d_pred, dim=1)
        _, pred_z = torch.max(d_pred_log_softmax, dim=1, keepdim=True)
        reward = d_pred_log_softmax[torch.arange(d_pred.shape[0]),
                                    z_hat] - math.log(1 / self.skill_dim)
        reward = reward.reshape(-1, 1)

        return reward * self.ours_scale

    def compute_ours_loss(self, trajectory, skill):
        """
        DF Loss
        """
        z_hat = torch.argmax(skill, dim=1)
        d_pred = self.ours(trajectory)
        d_pred_log_softmax = F.log_softmax(d_pred, dim=1)
        _, pred_z = torch.max(d_pred_log_softmax, dim=1, keepdim=True)
        d_loss = self.ours_criterion(d_pred, z_hat)
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
            trajectory = []
            for skil in skill.cpu().numpy():
                tau = random.sample(self.skills2taus[np.argmax(skil)], 1)
                trajectory.append(tau)
            trajectory = torch.as_tensor(np.asarray(trajectory), device=self.device)
            trajectory = trajectory.view(skill.shape[0], -1)

            metrics.update(self.update_ours(skill, trajectory, step))

            with torch.no_grad():
                intr_reward = self.compute_intr_reward(skill, trajectory, step)

            if self.use_tb or self.use_wandb:
                metrics['intr_reward'] = intr_reward.mean().item()
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
