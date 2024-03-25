#  Prior controlled diffusers Q learning

import os
import numpy as np
from agents.agent import Agent
from typing import Optional, Sequence, Tuple, Union, Callable
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from datasets.replay_buffer import Batch
from networks.types import InfoDict
EPS = 1e-6


def ema_update(src_model: nn.Module, target_model: nn.Module, ema_tau: float):
    with torch.no_grad():
        for src_p, tar_p in zip(src_model.parameters(), target_model.parameters()):
            tar_p.mul_(1.0 - ema_tau).add_(src_p.data, alpha=ema_tau)


class CNNCritic(nn.Module):
    """
    A simple critic network with CNN layers
    """

    def __init__(self,
                 input_dims: int,
                 hidden_dims: Sequence[int],
                 out_dims: int,
                 height: int,
                 width: int,
                 layer_norm: bool = False,
                 dropout: float = 0):
        super().__init__()

        conv = []
        for (in_d, out_d) in zip((input_dims,) + tuple(hidden_dims), hidden_dims):
            conv.append(nn.Conv2d(in_d, out_d, 3, padding=1))
            if layer_norm:
                conv.append(nn.LayerNorm([out_d, height, width]))
            conv.append(nn.Mish())
            if dropout is not None and dropout > 0:
                conv.append(nn.Dropout(p=dropout))

        self.conv = nn.Sequential(*conv)
        self.fc = nn.Linear(hidden_dims[-1] * height * width, out_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.fc(x.reshape(x.shape[0], -1))


class DQNLearner(Agent):
    name = "torch_dqn"  # Double DQN
    model_names = ["critic", "critic_tar"]

    def __init__(self,
                 obs_shape: tuple,  # (H, W, C)
                 act_dim: int,
                 critic_lr: float = 3e-4,
                 hidden_dims: Sequence[int] = (64, 64, 64, 1),
                 clip_grad_norm: float = 1,
                 layer_norm: bool = False,
                 dropout_rate: Optional[float] = None,
                 discount: float = 0.99,
                 greedy_max: float = 1,
                 greedy_min: float = 0.01,
                 greedy_decay_steps: float = 1000000,
                 ema_tau: float = 0.005,  # ema for critic learning
                 update_ema_every: int = 5,
                 step_start_ema: int = 1000,
                 lr_decay_steps: int = 2000000,
                 device: str = 'cpu',
                 **kwargs,
                 ):

        # models
        self.device = torch.device(device)
        self.act_dim = act_dim
        self.critic = CNNCritic(input_dims=obs_shape[2],
                                hidden_dims=hidden_dims,
                                out_dims=act_dim,
                                height=obs_shape[0],
                                width=obs_shape[1],
                                layer_norm=layer_norm,
                                dropout=dropout_rate).to(self.device)
        self.critic_tar = CNNCritic(input_dims=obs_shape[2],
                                    hidden_dims=hidden_dims,
                                    out_dims=act_dim,
                                    height=obs_shape[0],
                                    width=obs_shape[1],
                                    layer_norm=layer_norm,
                                    dropout=dropout_rate).to(self.device)

        self.critic.eval()
        self.critic_tar.eval()

        # training
        self.opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=lr_decay_steps)
        self.clip_grad_norm = clip_grad_norm

        # sampler
        self.discount = discount
        self.ema_tau = ema_tau
        self.update_ema_every = update_ema_every
        self.step_start_ema = step_start_ema

        self.eps_greedy_fn = lambda t: max(greedy_max - (greedy_max - greedy_min) * t / greedy_decay_steps, greedy_min)

        self.training = True
        self._n_training_steps = 0

    def update(self, batch: Batch) -> InfoDict:

        # Q-learning

        batch_size = batch.observations.shape[0]
        self.critic.train()

        # hwc to chw tensors
        chw_obs = np.transpose(batch.observations, (0, 3, 1, 2))
        chw_next_obs = np.transpose(batch.next_observations, (0, 3, 1, 2))
        obs = torch.FloatTensor(chw_obs).to(self.device)
        act = torch.LongTensor(batch.actions).to(self.device)
        next_obs = torch.FloatTensor(chw_next_obs).to(self.device)
        rewards = torch.FloatTensor(batch.rewards).to(self.device)
        masks = torch.FloatTensor(batch.masks).to(self.device)

        # double DQN
        with torch.no_grad():
            greedy_actions = self.critic(obs).argmax(axis=-1)
            next_qs = self.critic_tar(next_obs)[
                range(batch_size), greedy_actions]  # (B, act_dim) -> (B,)
            target_q = rewards + self.discount * masks * next_qs

        qs = self.critic(obs)[range(batch_size), act]
        tq_loss = ((qs - target_q) ** 2).mean()

        self.opt.zero_grad()
        tq_loss.backward()
        clip_grad_norm_(self.critic.parameters(), max_norm=self.clip_grad_norm, norm_type=2)
        self.opt.step()
        self.lr_scheduler.step()

        # EMA
        ema_update(self.critic, self.critic_tar, self.ema_tau)

        self.critic.eval()
        self._n_training_steps += 1
        return {
                'critic_loss': tq_loss.detach().item(),
                'qs': qs.mean().detach().item(),
            }

    def sample_actions(self,
                       observations: np.ndarray,  # (H, W, C)
                       legal_actions: np.ndarray,  # (act_dim, )  1 -> legal, 0 -> illegal
                       ) -> Union[int, np.ndarray]:

        observations = torch.FloatTensor(np.transpose(observations, (2, 0, 1))).to(self.device)

        # epsilon-greedy
        if self.training and np.random.uniform() < self.eps_greedy_fn(self._n_training_steps):
            action = np.random.choice(np.nonzero(legal_actions)[0])  # without batch support
        else:
            with torch.no_grad():
                # greedy actions  (B, act_dim)
                p = np.exp(self.critic_tar(observations[np.newaxis, :]) + EPS) * legal_actions[np.newaxis, :]
                action = p.argmax(axis=-1)[0]
        return int(action)  # [act] -> act

    def save_ckpt(self, prefix: str, ckpt_folder: str = "ckpt"):
        save_dir = os.path.join(ckpt_folder, self.name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for n_ in self.model_names:
            save_target = os.path.join(save_dir, prefix + n_)
            torch.save(self.__getattribute__(n_).state_dict(), save_target)

    def load_ckpt(self, prefix: str, ckpt_folder: str = "ckpt"):
        save_dir = os.path.join(ckpt_folder, self.name)
        for n_ in self.model_names:
            save_target = os.path.join(save_dir, prefix + n_)
            self.__getattribute__(n_).load_state_dict(torch.load(save_target))

        print(f"Successfully loaded checkpoint '{prefix}' from {save_dir}")

