from typing import Optional, Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp

from jaxrl2.networks import MLP
from jaxrl2.networks.constants import default_init


class LearnedStdNormalPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    dropout_rate: Optional[float] = None
    log_std_min: Optional[float] = -20
    log_std_max: Optional[float] = 2

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 training: bool = False) -> distrax.Distribution:
        outputs = MLP(self.hidden_dims,
                      activate_final=True,
                      dropout_rate=self.dropout_rate)(observations,
                                                      training=training)

        means = nn.Dense(self.action_dim, kernel_init=default_init(1e-2))(outputs)

        log_stds = nn.Dense(self.action_dim, kernel_init=default_init(1e-2))(outputs)
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds))
        return distribution

class TanhMultivariateNormalDiag(distrax.Transformed):

    def __init__(self,
                 loc: jnp.ndarray,
                 scale_diag: jnp.ndarray,
                 low: Optional[jnp.ndarray] = None,
                 high: Optional[jnp.ndarray] = None):
        distribution = distrax.MultivariateNormalDiag(loc=loc,
                                                      scale_diag=scale_diag)

        layers = []

        if not (low is None or high is None):

            def rescale_from_tanh(x):
                x = (x + 1) / 2  # (-1, 1) => (0, 1)
                return x * (high - low) + low

            def forward_log_det_jacobian(x):
                high_ = jnp.broadcast_to(high, x.shape)
                low_ = jnp.broadcast_to(low, x.shape)
                return jnp.sum(jnp.log(0.5 * (high_ - low_)), -1)

            layers.append(
                distrax.Lambda(
                    rescale_from_tanh,
                    forward_log_det_jacobian=forward_log_det_jacobian,
                    event_ndims_in=1,
                    event_ndims_out=1))

        layers.append(distrax.Block(distrax.Tanh(), 1))

        bijector = distrax.Chain(layers)

        super().__init__(distribution=distribution, bijector=bijector)

    def mode(self) -> jnp.ndarray:
        return self.bijector.forward(self.distribution.mode())

import distrax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Sequence

def make_tanh_multivariate_normal_diag(loc: jnp.ndarray,
                                       scale_diag: jnp.ndarray) -> distrax.Distribution:
    """
    loc, scale_diag: shape (B, D)
    반환: Transformed(Independent(Normal), Block(Tanh, ndims=1))
    """
    base = distrax.Normal(loc=loc, scale=scale_diag)                      # (B, D) elementwise
    event = distrax.Independent(base, reinterpreted_batch_ndims=1)        # 이벤트=마지막축(D)
    tanh_bij = distrax.Block(distrax.Tanh(), ndims=1)                     # 이벤트 축에만 tanh
    dist = distrax.Transformed(event, tanh_bij)                           # 최종 분

    return dist
   

class LearnedStdTanhNormalPolicy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int = 1600   
    dropout_rate: Optional[float] = None
    log_std_min: float = -20.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(self, observations: jnp.ndarray, training: bool = False):
        x = MLP(self.hidden_dims, activate_final=True, dropout_rate=self.dropout_rate)(
            observations, training=training
        )
        means = nn.Dense(self.action_dim, kernel_init=default_init(1e-2))(x)
        log_stds = nn.Dense(self.action_dim, kernel_init=default_init(1e-2))(x)
        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        dist = make_tanh_multivariate_normal_diag(
            loc=means, scale_diag=jnp.exp(log_stds)
        )
        return dist


# class LearnedStdTanhNormalPolicy(nn.Module):
#     hidden_dims: Sequence[int]
#     action_dim: int =  32
#     dropout_rate: Optional[float] = None
#     log_std_min: Optional[float] = -20
#     log_std_max: Optional[float] = 2
#     low: Optional[float] = None
#     high: Optional[float] = None

#     @nn.compact
#     def __call__(self,
#                  observations: jnp.ndarray,
#                  training: bool = False) -> distrax.Distribution:
        
#         outputs = MLP(self.hidden_dims,
#                       activate_final=True,
#                       dropout_rate=self.dropout_rate)(observations,
#                                                       training=training)

#         means = nn.Dense(self.action_dim, kernel_init=default_init(1e-2))(outputs)

#         log_stds = nn.Dense(self.action_dim, kernel_init=default_init(1e-2))(outputs)
#         log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

#         distribution = TanhMultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds), low=self.low, high=self.high)
#         return distribution