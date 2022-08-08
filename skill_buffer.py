import numpy as np
import torch


class SkillsBuffer:
    """
    A simple FIFO experience replay buffer to storing B(tau | z).
    """

    def __init__(self, tau_dim, batch_size, skill_dim, action_dim, tau_len):
        self.skills2taus = np.zeros((skill_dim, batch_size, tau_dim), dtype=np.float32)
        self._ptrs, self.sizes, self.max_size = [0] * skill_dim, [0] * skill_dim, batch_size
        self.action_dim = action_dim
        # self._discount = discount
        self.skills2actions = np.zeros((skill_dim, batch_size, action_dim * tau_len), dtype=np.float32)
        # self.skills2discount = np.zeros((skill_dim, batch_size, tau_len), dtype=np.float32)

    def store(self, skill, tau, actions):
        taus = self.skills2taus[skill]
        taus[self._ptrs[skill]] = tau
        self.skills2actions[skill][self._ptrs[skill]] = actions
        # self.skills2discount[skill][self._ptrs[skill]] = discount
        self._ptrs[skill] = (self._ptrs[skill] + 1) % self.max_size
        self.sizes[skill] = min(self.sizes[skill] + 1, self.max_size)

    def wapper(self, data, size):
        return torch.as_tensor(data).view(size, -1)

    def sample_batch(self, batch_size, counts=None):
        # 'skills' is scalar or count bucket
        rows = np.asarray([], dtype=np.int32)
        cols = np.asarray([], dtype=np.int32)
        for skill, batch_size_per_skill in enumerate(counts):
            if batch_size_per_skill == 0:
                continue
            col = np.random.randint(0, self.sizes[skill], size=batch_size_per_skill)
            rows = np.concatenate([rows, np.asarray([skill] * batch_size_per_skill, dtype=np.int32)])
            cols = np.concatenate([cols, col])

        tau_batch = self.skills2taus[rows, cols]
        actions_batch = self.skills2actions[rows, cols]

        # tau_batch.shape = batch_size, obs_dim * tau_len
        # actions_batch.shape = batch_size, action_dim * tau_len
        return self.wapper(tau_batch, batch_size), self.wapper(actions_batch, batch_size)

