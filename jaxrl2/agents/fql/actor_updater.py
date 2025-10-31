from typing import Dict, Tuple
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from jaxrl2.data.dataset import DatasetDict
from jaxrl2.types import Params, PRNGKey
from examples.train_utils_sim import batch_size

# H, DA_ACT = 50, 14
# AH_ACT = H * DA_ACT 

def update_actor(key, actor, critic, batch, critic_reduction='mean', alpha=1.0, normalize_q_loss=False):
    bsz = batch['observations']['pixels'].shape[0]

    def loss_fn(actor_params):
        vars_act = {'params': actor_params}
        if getattr(actor, 'batch_stats', None) is not None:
            vars_act['batch_stats'] = actor.batch_stats
        dist = actor.apply_fn(vars_act, batch['observations'], training=True)

        #  mode() 쓰지 말고 샘플링
        rng_act, subkey = jax.random.split(key)
        actions, logp = dist.sample_and_log_prob(seed=subkey)   # (B,1600), (B,)

        # ─ Distill (옵션1: (B,1600) 이벤트) ─
        if 'pi0_noises' in batch:
            w_tgt = batch['pi0_noises'].reshape(bsz, -1).astype(jnp.float32)
            eps = 1e-6
            w_tgt = jnp.clip(w_tgt, -1.0 + eps, 1.0 - eps)
            # 분포 래핑을 Fix A로 만들었다면 그냥 사용:
            log_probs = dist.log_prob(w_tgt)                   # (B,)
            # (우회가 필요하면 atanh+야코비안 보정 방식 사용)
            distill = (-log_probs).mean()
        else:
            distill = jnp.asarray(0.0, actions.dtype)

        # ─ Q-term ─
        qs = critic.apply_fn({'params': critic.params}, batch['observations'], actions)
        q = qs.min(axis=0) if critic_reduction == 'min' else qs.mean(axis=0)
        q_loss = (-q).mean()

        loss = q_loss + alpha * distill
        return loss, {'actor_loss': loss, 'q_loss': q_loss, 'distill_loss': distill}


    grads, info = jax.grad(loss_fn, has_aux=True)(actor.params)
    new_actor = actor.apply_gradients(grads=grads)
    return new_actor, info

