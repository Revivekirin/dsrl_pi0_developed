import time
import jax
import numpy as np

from flax.traverse_util import flatten_dict

def _now_ms():
    return int(time.perf_counter() * 1000)

class DebugTimer:
    def __init__(self, prefix="[DEBUG]"):
        self.prefix = prefix
        self.stack = []

    def step(self, name):
        self.stack.append((name, _now_ms()))
        print(f"{self.prefix} ▶ {name} ...")
        return name  # no-op for 'with' if not using contextmanager

    def end(self, name=None, extra_msg=""):
        if not self.stack:
            return
        sname, t0 = self.stack.pop()
        if name is None:
            name = sname
        dt = _now_ms() - t0
        print(f"{self.prefix} ✓ {name} took {dt} ms {extra_msg}")

def debug_shape(name, arr):
    try:
        shape = tuple(arr.shape)
        dtype = getattr(arr, 'dtype', None)
    except Exception:
        shape, dtype = "<?>", None
    print(f"[DEBUG] {name}: shape={shape} dtype={dtype}")

def debug_sync_tree(tree, label="sync"):
    """JAX lazy exec을 강제 동기화하여 실제 compute 시간을 측정."""
    t0 = _now_ms()
    def _block(x):
        if hasattr(x, "block_until_ready"):
            x.block_until_ready()
        return x
    jax.tree_util.tree_map(_block, tree)
    dt = _now_ms() - t0
    print(f"[DEBUG] sync '{label}' took {dt} ms")



def debug_pytree_shapes(name, tree, limit=None):
    """
    FrozenDict / PyTree의 모든 leaf에 대해 shape, dtype 출력.
    limit: 출력 항목 개수 제한(너무 많을 때)
    """
    try:
        flat = flatten_dict(tree, sep='/')  # {'observations/pixels': ndarray, ...}
    except Exception:
        flat = {}

    rows = []
    def _leaf_info(x):
        shape = getattr(x, 'shape', None)
        dtype = getattr(x, 'dtype', None)
        return shape, dtype

    if flat:
        for k, v in flat.items():
            shape, dtype = _leaf_info(v)
            rows.append((k, shape, dtype))
    else:
        leaves, treedef = jax.tree_util.tree_flatten(tree)
        for i, v in enumerate(leaves):
            shape, dtype = _leaf_info(v)
            rows.append((f'leaf[{i}]', shape, dtype))

    print(f"[DEBUG] {name} pytree leaves: {len(rows)}")
    if limit is not None:
        rows = rows[:limit]
    for k, shape, dtype in rows:
        print(f"[DEBUG]  - {k}: shape={shape}, dtype={dtype}")
