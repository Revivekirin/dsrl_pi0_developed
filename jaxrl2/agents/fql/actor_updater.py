from audioop import cross
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from jaxrl2.data.dataset import DatasetDict
from jaxrl2.types import Params, PRNGKey

from examples.train_utils_sim import batch_size

def update_actor(
    key: PRNGKey,
    actor: TrainState,
    critic: TrainState,
    batch: DatasetDict,
    *,
    distill_noises: jnp.ndarray,      
    distill_targets: jnp.ndarray,     
    critic_reduction: str = 'min',
    alpha_distill: float = 1.0,
    use_bc_flow: bool = False,
    normalize_q_loss: bool = False,
) -> Tuple[TrainState, Dict[str, float]]:

    bsz = batch_size(batch["actions"])
    Da = Da = int(batch['actions'].shape[-2] * batch['actions'].shape[-1])

    # one-step policy -> Fixed t (1.0)
    t = jnp.ones((bsz, 1), dtype=jnp.float32) 

    def actor_loss_fn(actor_params: Params):
        vars_act = {'params': actor_params}
        if getattr(actor, 'batch_stats', None) is not None:
            vars_act['batch_stats'] = actor.batch_stats

        a_student = actor.apply_fn(vars_act, batch['observations'], distill_noises, t, True)  # (bsz, Da)

        # ----- Distillation loss (to π₀) -----
        distill_loss = jnp.mean((a_student - jax.lax.stop_gradient(distill_targets))**2)

        # ----- Q term -----
        vars_cri = {'params': critic.params}
        if getattr(critic, 'batch_stats', None) is not None:
            vars_cri['batch_stats'] = critic.batch_stats

        qs = critic.apply_fn(vars_cri, batch['observations'], a_student, False)  # (num_q, bsz) or (bsz,)
        q  = qs.min(axis=0) if critic_reduction == 'min' else qs.mean(axis=0)
        q_loss = (-q).mean()
        if normalize_q_loss:
            lam = jax.lax.stop_gradient(1.0 / (jnp.abs(q).mean() + 1e-8))
            q_loss = lam * q_loss

        # -----  BC flow-matching -----
        bc_flow_loss = 0.0
        if use_bc_flow:
            key1, key2 = jax.random.split(key)
            x0 = jax.random.normal(key1, (bsz, Da))
            x1 = batch['actions'].reshape(bsz, -1)     
            t_b = jax.random.uniform(key2, (bsz, 1))  #TODO:time embedding 수정
            xt = (1 - t_b) * x0 + t_b * x1
            vel = x1 - x0
            v_pred = actor.apply_fn(vars_act, batch['observations'], xt, t_b, True,
                                    method=getattr(actor.apply_fn.__self__.network, 'vf'))  # 구현에 맞게 수정
            bc_flow_loss = jnp.mean((v_pred - vel)**2)

        loss = q_loss + alpha_distill * distill_loss + bc_flow_loss
        info = {
            'actor_loss': loss,
            'q_loss': q_loss,
            'distill_loss': distill_loss,
            'bc_flow_loss': bc_flow_loss,
            'q_mean': q.mean(),
            'action_norm': jnp.linalg.norm(a_student, axis=-1).mean(),
        }
        return loss, info

    grads, info = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    new_actor = actor.apply_gradients(grads=grads)
    return new_actor, info