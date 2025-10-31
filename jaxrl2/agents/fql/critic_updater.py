from typing import Dict, Tuple
import numpy as np

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from jaxrl2.data.dataset import DatasetDict
from jaxrl2.types import Params, PRNGKey
from examples.train_utils_sim import batch_size


H = 50
def update_critic(key, actor, critic, target_critic, batch, discount, critic_reduction='min'):
    next_qs = target_critic.apply_fn({'params': target_critic.params},
                                     batch['next_observations'], batch['next_actions'])
    
    if critic_reduction == 'min':
        next_q = next_qs.min(axis=0)
    elif critic_reduction == 'mean':
        next_q = next_qs.mean(axis=0)
    else:
        raise NotImplemented()


    r_chunk = batch['rewards']         
    m_chunk = batch.get('masks', jnp.ones_like(r_chunk))  # (B,)

    target_q = r_chunk + (discount ** H) * m_chunk * next_q  # (B,)

    def critic_loss_fn(critic_params):
        cri_vars = {'params': critic_params}
        if getattr(critic, 'batch_stats', None) is not None:
            cri_vars['batch_stats'] = critic.batch_stats
  
        dist = actor.apply_fn({'params': actor.params}, batch['next_observations'])
        next_actions, next_log_probs = dist.sample_and_log_prob(seed=key) 
        qs = critic.apply_fn(cri_vars, batch['observations'], next_actions, False)  
        q = qs.mean(axis=0) if qs.ndim == 2 else qs
        loss = ((q - target_q) ** 2).mean()
        info = {
            'critic_loss': loss,
            'q_mean': q.mean(),
            'target_q_mean': target_q.mean(),
            'next_q_mean': next_q.mean(),
        }
        return loss, info

    grads, info = jax.grad(critic_loss_fn, has_aux=True)(critic.params)
    new_critic = critic.apply_gradients(grads=grads)
    return new_critic, info
