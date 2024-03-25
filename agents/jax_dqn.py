#  Prior controlled diffusers Q learning

import numpy as np
from functools import partial
from typing import Optional, Sequence, Tuple, Union, Callable
import os
import orbax
import jax
import jax.numpy as jnp
import optax
from flax.training import orbax_utils
from networks.model import Model
from networks.nets import CNN, mish
from networks.updates import ema_update
from datasets.replay_buffer import Batch
from networks.types import InfoDict, Params, PRNGKey
from agents.agent import Agent


EPS = 1e-6


@partial(jax.jit, static_argnames=('discount', 'ema_tau'))
def _jit_update_critic(critic: Model,
                       critic_tar: Model,
                       discount: float,
                       ema_tau: float,
                       batch: Batch,
                       ):
    # rng, = jax.random.split(rng)

    batch_size = batch.observations.shape[0]

    # double DQN
    greedy_actions = critic(batch.observations).argmax(axis=-1)
    next_qs = critic_tar(batch.next_observations)[jnp.arange(batch_size), greedy_actions]  # (B, act_dim) -> (B,)
    target_q = batch.rewards + discount * batch.masks * next_qs

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        qs = critic.apply(critic_params, batch.observations, )[jnp.arange(batch_size), batch.actions]
        critic_loss = ((qs - target_q) ** 2).mean()
        return critic_loss, {
            'critic_loss': critic_loss,
            'qs': qs.mean(),
        }

    new_critic, info = critic.apply_gradient(critic_loss_fn)
    new_target_critic = ema_update(new_critic, critic_tar, ema_tau)

    return new_critic, new_target_critic, info


@partial(jax.jit, static_argnames=('critic_tar_apply_fn',))
def _jit_greedy_actions(rng: PRNGKey,
                        critic_tar_apply_fn: Callable,
                        critic_tar_params: Params,
                        observations: jnp.ndarray,
                        legal_actions: jnp.ndarray,  # (B, act_dim)
                        ) -> [PRNGKey, jnp.ndarray]:
    rng, key = jax.random.split(rng)

    # greedy actions  (B, act_dim)
    p = (jnp.exp(critic_tar_apply_fn(critic_tar_params, observations)) + EPS) * legal_actions
    action = p.argmax(axis=-1)

    # used for evaluation
    return rng, action[0]


class DQNLearner(Agent):
    name = "jax_dqn"  # Double DQN
    model_names = ["critic", "critic_tar"]

    def __init__(self,
                 obs_shape: tuple,
                 act_dim: int,
                 seed: int,
                 critic_lr: Union[float, optax.Schedule] = 3e-4,
                 hidden_dims: Sequence[int] = (64, 64, 64, 1),
                 clip_grad_norm: float = 1,
                 dropout_rate: Optional[float] = None,
                 layer_norm: bool = False,
                 discount: float = 0.99,
                 greedy_max: float = 1,
                 greedy_min: float = 0.01,
                 greedy_decay_steps: float = 1000000,
                 ema_tau: float = 0.005,  # ema for critic learning
                 update_ema_every: int = 5,
                 step_start_ema: int = 1000,
                 lr_decay_steps: int = 2000000,
                 **kwargs,
                 ):

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key = jax.random.split(rng, 4)

        if lr_decay_steps is not None:
            critic_lr = optax.cosine_decay_schedule(critic_lr, lr_decay_steps)

        critic_def = CNN(hidden_dims=tuple(hidden_dims),
                         out_dim=act_dim,
                         activations=mish,
                         dropout_rate=dropout_rate,
                         layer_norm=layer_norm)

        critic = Model.create(critic_def,
                              inputs=[critic_key, jnp.empty(obs_shape)[jnp.newaxis, :]],
                              optimizer=optax.adam(learning_rate=critic_lr),
                              clip_grad_norm=clip_grad_norm)
        critic_tar = Model.create(critic_def,
                                  inputs=[critic_key, jnp.empty(obs_shape)[jnp.newaxis, :]])
        # models
        self.act_dim = act_dim
        self.critic = critic
        self.critic_tar = critic_tar

        # sampler
        self.discount = discount
        self.ema_tau = ema_tau
        self.update_ema_every = update_ema_every
        self.step_start_ema = step_start_ema

        self.eps_greedy_fn = lambda t: max(greedy_max - (greedy_max - greedy_min)*t/greedy_decay_steps, greedy_min)

        self.rng = rng
        self.training = True
        self._n_training_steps = 0

    def update(self, batch: Batch) -> InfoDict:

        # Q-learning
        self.critic, self.critic_tar, info = _jit_update_critic(self.critic,
                                                                self.critic_tar,
                                                                self.discount,
                                                                self.ema_tau,  # ema
                                                                batch,
                                                                )

        self._n_training_steps += 1
        return info

    def sample_actions(self,
                       observations: np.ndarray,  # (H, W, C)
                       legal_actions: np.ndarray,  # (act_dim, ) 1 -> legal, 0 -> illegal
                       ) -> Union[int, jnp.ndarray]:

        observations = jax.device_put(observations)
        self.rng, key = jax.random.split(self.rng)

        # epsilon-greedy
        if self.training and jax.random.uniform(key) < self.eps_greedy_fn(self._n_training_steps):
            self.rng, key = jax.random.split(self.rng)
            action = jax.random.choice(key, np.nonzero(legal_actions)[0])  # without batch support
            # action = jax.random.randint(key, (observations.shape[0],), 0, self.act_dim)
        else:
            self.rng, action = _jit_greedy_actions(self.rng,
                                                   self.critic_tar.apply,
                                                   self.critic_tar.params,
                                                   observations[np.newaxis, :],  # jax always requires the batch-dim
                                                   legal_actions[np.newaxis, :])
        return action

    def save_ckpt(self, prefix: str, ckpt_folder: str = "ckpt", silence: bool = True,
                  legacy: bool = False):
        """
        load & save models by the model attribute names
        """
        assert prefix
        # save_dir = os.path.join(ckpt_folder, self.name, prefix)
        save_dir = os.path.join(ckpt_folder, self.name)
        for n_ in self.model_names:
            save_target = os.path.join(save_dir, prefix + n_)
            if not legacy:
                orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
                ckpt = {'model': self.__getattribute__(n_)}
                save_args = orbax_utils.save_args_from_target(ckpt)
                orbax_checkpointer.save(save_target, ckpt, save_args=save_args, force=True)

            else:
                self.__getattribute__(n_).save(save_target)

            if not silence:
                print(f"Successfully save {n_} model to {save_target}")

    def load_ckpt(self, prefix: str = "", ckpt_folder: str = "ckpt", silence: bool = False,
                  legacy: bool = False):
        # save_dir = os.path.join(ckpt_path, self.name, prefix)
        assert prefix
        save_dir = os.path.join(ckpt_folder, self.name)
        for n_ in self.model_names:
            restore_target = os.path.join(save_dir, prefix + n_)
            if not legacy:
                orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
                target = {'model': self.__getattribute__(n_)}  # use raw as the tree prototype
                restore_model = orbax_checkpointer.restore(restore_target, item=target)
                self.__setattr__(n_, restore_model['model'])
            else:
                self.__setattr__(n_, self.__getattribute__(n_).load(restore_target))
            if not silence:
                print(f"Successfully load {n_} model from {restore_target}")
