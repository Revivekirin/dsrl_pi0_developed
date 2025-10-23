from typing import Dict, Tuple
import numpy as np

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from jaxrl2.data.dataset import DatasetDict
from jaxrl2.types import Params, PRNGKey
from examples.train_utils_sim import batch_size


def update_critic(
        key: PRNGKey, actor: TrainState, critic: TrainState,
        target_critic: TrainState,  batch: DatasetDict,
        discount: float, action_dim: int, 
        critic_reduction: str = 'min') -> Tuple[TrainState, Dict[str, float]]:
    #TODO:  next actions sampling from actor_updater 
    # dist = actor.apply_fn({'params': actor.params}, batch['next_observations'])
    # next_actions, next_log_probs = dist.sample_and_log_prob(seed=key)

    next_obs = batch['next_observations']
    obs = batch['observations']
    bsz = batch_size(batch["observations"])
    Da = int(np.prod(batch["actions"].shape[-2:]))

    # one-step policy -> Fixed t (1.0)
    t_next = jnp.ones((bsz, 1), dtype=jnp.float32)
    noises = jax.random.normal(key, (bsz, Da), dtype=jnp.float32)
    actor_vars = {'params' : actor.params}
    
    next_actions = actor.apply_fn(actor_vars, next_obs, noises, t_next, False) #TODO : time embedding 고려
  
    next_qs = target_critic.apply_fn({'params': target_critic.params},
                                     batch['next_observations'], next_actions)
    if critic_reduction == 'min':
        next_q = next_qs.min(axis=0)
    elif critic_reduction == 'mean':
        next_q = next_qs.mean(axis=0)
    else:
        raise NotImplemented()

    target_q = batch['rewards'] + batch["discount"] * batch['masks'] * next_q

    def critic_loss_fn(critic_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        critic_vars = {'params': critic_params}
        if getattr(critic, "batch_stats", None) is not None:
            critic_vars['batch_stats'] = critic.batch_stats

        # critic.apply_fn(observations, actions, training=False)
        qs = critic.apply_fn(critic_vars, batch['observations'], batch['actions'], False)

        loss = jnp.mean((qs - target_q) ** 2)

        info = {
            'critic_loss': loss,
            'q_mean': qs.mean(),
            'q_max': qs.max(),
            'q_min': qs.min(),
            'target_q_mean': target_q.mean(),
            'next_q_mean': next_q.mean(),
        }
        return loss, info

    grads, info = jax.grad(critic_loss_fn, has_aux=True)(critic.params)
    new_critic = critic.apply_gradients(grads=grads)

    return new_critic, info