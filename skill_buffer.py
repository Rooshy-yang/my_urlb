import numpy as np
import torch


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class SkillsBuffer:
    """
    A simple FIFO experience replay buffer to storing B(tau | z).
    """

    def __init__(self, tau_dim, size, skill_dim):
        self.skills2taus = np.zeros((skill_dim, size, tau_dim), dtype=np.float32)
        self._ptrs, self.sizes, self.max_size = [0] * skill_dim, [0] * skill_dim, size

    def store(self, skill, tau):
        taus = self.skills2taus[skill]
        taus[self._ptrs[skill]] = tau
        self._ptrs[skill] = (self._ptrs[skill] + 1) % self.max_size
        self.sizes[skill] = min(self.sizes[skill] + 1, self.max_size)

    def wapper(self, data, size):
        return torch.as_tensor(data).view(size, -1)

    def sample_batch(self, batch_size, skills=None, counts=None):
        # 'skills' is scalar or count bucket
        if counts is None:
            idxs = np.random.randint(0, self.sizes[skills], size=batch_size)
            batch = self.skills2taus[skills][idxs]
        else:
            rows = np.asarray([], dtype=np.int32)
            cols = np.asarray([], dtype=np.int32)
            for skill, batch_size_per_skill in enumerate(counts):
                if batch_size_per_skill == 0:
                    continue
                col = np.random.randint(0, self.sizes[skill], size=batch_size_per_skill)
                rows = np.concatenate([rows, np.asarray([skill] * batch_size_per_skill, dtype=np.int32)])
                cols = np.concatenate([cols, col])
            batch = self.skills2taus[rows, cols]

        return self.wapper(batch, batch_size)

