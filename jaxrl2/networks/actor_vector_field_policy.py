# vector_field.py
from typing import Optional, Sequence
import flax.linen as nn
import jax.numpy as jnp

from jaxrl2.networks import MLP  


class OneStepFlowActor(nn.Module): # step 마다 v에 따른 action 값 출력 
    """z ~ N(0,I) + obs -> action.
       a = clip( z + alpha * v_theta(o, z[, t]) , [-1,1] )
    """
    hidden_dims: Sequence[int]
    action_dim: int
    alpha: float = 1.0        # step size
    layer_norm: bool = False
    use_tanh_clip: bool = True  # [-1,1] or not

    def setup(self):
        self.vf = ActorVectorField(self.hidden_dims, self.action_dim,
                                   layer_norm=self.layer_norm)

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 noises: jnp.ndarray,
                 times: Optional[jnp.ndarray] = None,
                 training: bool = False):
        
        v = self.vf(observations, noises, times=times, training=training)
        a = noises + self.alpha * v
        if self.use_tanh_clip:
            a = jnp.tanh(a)  # [-1,1]
        return a


def _concat(*xs):
    xs = [x for x in xs if x is not None]
    return jnp.concatenate(xs, axis=-1)


class ActorVectorField(nn.Module):
    """v_theta(o, a[, t])"""
    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False

    def setup(self):
        self.mlp = MLP((*self.hidden_dims, self.action_dim),
                       activate_final=False,
                       use_layer_norm=self.layer_norm)

    @nn.compact
    def __call__(self,
                 observations: jnp.ndarray,
                 actions: jnp.ndarray,
                 times: Optional[jnp.ndarray] = None,
                 training: bool = False):
        """
        Args:
          observations: (..., Do)
          actions:      (..., Da)  
          times:        (..., 1)  
        Returns:
          v: (..., Da)  
        """
        h = _concat(observations, actions, times)
        v = self.mlp(h, training=training)
        return v
