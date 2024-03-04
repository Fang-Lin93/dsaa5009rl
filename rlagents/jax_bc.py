from typing import Tuple, Sequence, Any, Optional, Callable
import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax
import functools
from flax import struct
import flax.linen as nn
from abc import ABC
from rlagents.agent import Agent, InfoDict
from dataset import Batch


"""Implementations of algorithms for continuous control."""
Params = flax.core.FrozenDict[str, Any]


@struct.dataclass
class Model(struct.PyTreeNode, ABC):
    step: int
    network: nn.Module = flax.struct.field(pytree_node=False)
    params: Params
    optimizer: Optional[optax.GradientTransformation] = flax.struct.field(
        pytree_node=False)
    opt_state: Optional[optax.OptState] = None
    """
    its network defines the computation tree for the inputs, it's a skeleton
    its params is an independent object
    its optimizer is defined from 'import optax | optax.adam'
    its forward/backward functions are conducted using the customized apply_gradient function
    use self.replace function modifies the attributes
    step: record the number of gradient updates
    Model is initialized with Model.create() method
    """

    @classmethod
    def create(cls,
               network: nn.Module,
               inputs: Sequence[jnp.ndarray],  # sample of inputs
               optimizer: Optional[optax.GradientTransformation] = None,
               clip_grad_norm: float = None) -> 'Model':
        params = network.init(*inputs)  # (rng, other_inputs), params = {"params": ...}

        if optimizer is not None:
            if clip_grad_norm:
                optimizer = optax.chain(
                    optax.clip_by_global_norm(max_norm=clip_grad_norm),
                    optimizer)
            opt_state = optimizer.init(params)
        else:
            opt_state = None

        return cls(step=1,
                   network=network,
                   params=params,
                   optimizer=optimizer,
                   opt_state=opt_state
                   )

    def __call__(self, *args, **kwargs):
        # the network defined by the jax nn.Model should be used by apply function with {'params': P} and other ..
        return self.network.apply(self.params, *args, **kwargs)

    def apply(self, *args, **kwargs):
        return self.network.apply(*args, **kwargs)

    def apply_gradient(self, loss_fn) -> Tuple['Model', InfoDict]:
        grad_fn = jax.grad(loss_fn, has_aux=True)  # here auxiliary data is just the info dict
        grads, info = grad_fn(self.params)

        updates, new_opt_state = self.optimizer.update(grads, self.opt_state,
                                                       self.params)

        new_params = optax.apply_updates(self.params, updates)

        return self.replace(step=self.step + 1,
                            params=new_params,
                            opt_state=new_opt_state), info  # gives new model, info


@jax.jit
def _mse_update(actor: Model, batch: Batch) -> Tuple[Model, InfoDict]:
    def loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        actions = actor.apply(actor_params,
                              batch.observations)
        actor_loss = ((actions - batch.actions) ** 2).mean()
        return actor_loss, {'actor_loss': actor_loss}
    return actor.apply_gradient(loss_fn)


@functools.partial(jax.jit, static_argnames=('net_apply_fn',))
def jit_sample_actions(net_apply_fn: Callable,
                       params: Params,
                       observations: np.ndarray) -> jnp.ndarray:
    return net_apply_fn(params, observations)


class JAXBCLearner(Agent):
    def __init__(self,
                 seed: int,
                 obs_dim: int,
                 act_dim: int,
                 actor_lr: float = 3e-4,
                 layer_norm: bool = True,
                 hidden_dims: Sequence[int] = (256, 256, 256),
                 lr_decay_T: int = -1,
                 max_grad_norm: float = 1):
        rng = jax.random.PRNGKey(seed)
        self.rng, actor_key = jax.random.split(rng)

        # create actor network
        layers = []
        for out_c in hidden_dims:
            layers.append(nn.Dense(out_c, kernel_init=nn.initializers.xavier_uniform()))
            if layer_norm:
                layers.append(nn.LayerNorm(out_c))
            layers.append(nn.relu)
        layers.append(nn.Dense(act_dim, kernel_init=nn.initializers.xavier_uniform()))
        layers.append(nn.tanh)

        if lr_decay_T > 0:
            actor_lr = optax.cosine_decay_schedule(actor_lr, lr_decay_T)
        optimizer = optax.adam(learning_rate=actor_lr)

        self.actor = Model.create(nn.Sequential(layers),
                                  inputs=[actor_key, jnp.empty((1, obs_dim))],
                                  optimizer=optimizer,
                                  clip_grad_norm=max_grad_norm)

    def update(self, batch: Batch) -> InfoDict:
        self.actor, info = _mse_update(self.actor, batch)
        return info

    def sample_actions(self, observations: np.ndarray) -> jnp.ndarray:
        return jit_sample_actions(self.actor.apply, self.actor.params, observations)


