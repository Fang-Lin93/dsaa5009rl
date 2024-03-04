import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from typing import Sequence, Optional

from dataset import Batch
from rlagents.agent import Agent, InfoDict


class TorchBCLearner(Agent):
    """
    PyTorch's implementation of a simple behavior cloning agent
    """
    def __init__(self,
                 obs_dim: int,
                 act_dim: int,
                 actor_lr: float = 1e-3,
                 layer_norm: bool = True,
                 hidden_dims: Sequence[int] = (256, 256, 256),
                 lr_decay_T: int = -1,
                 max_grad_norm: float = 1,
                 device: torch.device = torch.device('cpu')):

        self.device = torch.device(device)

        # create actor network
        in_c = obs_dim
        layers = []
        for out_c in hidden_dims:
            layers.append(nn.Linear(in_c, out_c))
            if layer_norm:
                layers.append(nn.LayerNorm(out_c))
            layers.append(nn.ReLU())
            in_c = out_c
        layers.append(nn.Linear(in_c, act_dim))
        layers.append(nn.Tanh())

        self.actor = nn.Sequential(*layers).to(self.device)
        self.opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.lr_decay_T = lr_decay_T
        self.grad_norm = max_grad_norm
        if lr_decay_T > 0:
            self.actor_lr_scheduler = CosineAnnealingLR(self.opt, T_max=lr_decay_T, eta_min=0.)

    def update(self, batch: Batch) -> InfoDict:
        self.actor.train()
        self.opt.zero_grad()
        obs = torch.FloatTensor(batch.observations).to(self.device)
        actions = torch.FloatTensor(batch.actions).to(self.device)
        pred_act = self.actor(obs)
        loss = F.mse_loss(pred_act, actions)
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm)
        self.opt.step()

        if self.lr_decay_T > 0:
            self.actor_lr_scheduler.step()

        return {'loss': loss.item()}

    def sample_actions(self, observations: np.ndarray) -> np.ndarray:
        obs_tensor = torch.FloatTensor(observations).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            pred_act = self.actor(obs_tensor)
            return pred_act.detach().cpu().numpy()









