from typing import Optional, Union

import numpy as np
import collections

Batch = collections.namedtuple(
    'Batch',
    ['observations', 'actions', 'rewards', 'masks', 'next_observations'])


def sample_n_k(n, k):
    """Sample k distinct elements uniformly from range(n)"""
    """it is faster to get replace=False"""

    if not 0 <= k <= n:
        raise ValueError("Sample larger than population or is negative")
    if k == 0:
        return np.empty((0,), dtype=np.int64)
    elif 3 * k >= n:
        return np.random.choice(n, k, replace=False)
    else:
        result = np.random.choice(n, 2 * k)
        selected = set()
        selected_add = selected.add
        j = k
        for i in range(k):
            x = result[i]
            while x in selected:
                x = result[i] = result[j]
                j += 1
                if j == 2 * k:
                    # This is slow, but it rarely happens.
                    result[k:] = np.random.choice(n, k)
                    j = k
            selected_add(x)
        return result[:k]


class ReplayBuffer(object):
    """
    Discrete-action replay buffer: actions are integers of finite values
    """

    def __init__(self, obs_shape: tuple, capacity: int, rotation_expand_k: int = 0):

        self.capacity = capacity
        self.observations = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.empty((capacity,), dtype=np.int32)
        self.rewards = np.empty((capacity,), dtype=np.float32)
        self.masks = np.empty((capacity,), dtype=np.float32)  # = 1 - done
        self.next_observations = np.empty((capacity, *obs_shape), dtype=np.float32)

        self.size = 0
        self.insert_index = 0
        self.capacity = capacity
        self.rotation_expand_k = rotation_expand_k

    def insert(self, observation: np.ndarray, action: int,
               reward: float, mask: float, next_observation: np.ndarray):

        # expand by rotations
        for _ in range(1 + self.rotation_expand_k):
            self.observations[self.insert_index] = observation
            self.actions[self.insert_index] = action  # action indicator
            self.rewards[self.insert_index] = reward
            self.masks[self.insert_index] = mask
            self.next_observations[self.insert_index] = next_observation

            self.insert_index = (self.insert_index + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

            # rotation expansion: obs and actions
            observation, next_observation = np.rot90(observation), np.rot90(next_observation)
            row, col = action // 6, action % 6
            action = 6 * (5 - col) + row

    def sample(self, batch_size: int) -> Union[Batch, None]:
        if self.size < batch_size:
            return
        indices = sample_n_k(self.size, batch_size)
        return Batch(observations=self.observations[indices],
                     actions=self.actions[indices],
                     rewards=self.rewards[indices],
                     masks=self.masks[indices],
                     next_observations=self.next_observations[indices],)

    def check(self):
        # before check: set the environment to be deterministic
        for idx in range(self.size):
            if self.masks[idx] > 0:
                assert (self.observations[idx][0] +
                        self.actions[idx].reshape(6, 6) == self.next_observations[idx][0]).all(), \
                    f"index={idx}: action error"
            else:
                if self.rewards[idx] != 0:
                    board = self.next_observations[idx][0]
                    winner = 1
                    for i in range(6):
                        # row or column
                        if any(sum(board[i, j:j + 4]) >= 4 for j in range(3)) or any(
                                sum(board[j:j + 4, i]) >= 4 for j in range(3)):
                            winner = 0
                        # diagonal
                        if i < 3:
                            if sum(board[i + j, i + j] for j in range(4)) >= 4 or sum(
                                    board[i + j, 5 - j - i] for j in range(4)) >= 4:
                                winner = 0
                    assert (winner > 0) ^ (self.rewards[idx] > 0), f"index={idx}: winner-reward assignment error"  # XOR

        print("check passed!")
