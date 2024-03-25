
import flax.linen as nn
import jax.numpy as jnp
from typing import Callable, Optional, Sequence
from networks.initialization import default_init


def mish(x):
    return x * jnp.tanh(nn.softplus(x))


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    layer_norm: bool = False
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=default_init())(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:  # hidden layers
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.activations(x)
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training)
        return x


class CNN(nn.Module):
    hidden_dims: Sequence[int]
    out_dim: int
    activations: Callable[[jnp.ndarray], jnp.ndarray] = nn.relu
    activate_final: int = False
    layer_norm: bool = False
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        for i, size in enumerate(self.hidden_dims):
            x = nn.Conv(size, kernel_size=(3, 3), kernel_init=default_init())(x)  # (B, H, W, C)
            if i + 1 < len(self.hidden_dims) or self.activate_final:  # hidden layers
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.activations(x)
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x, deterministic=not training)

        # always requires the batch dim
        x = nn.Dense(self.out_dim, kernel_init=default_init())(x.reshape(x.shape[0], -1))
        return x
