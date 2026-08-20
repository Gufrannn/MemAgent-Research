from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # CPU-only repository checks may not install the training stack.
    torch = None

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "gate_a_weight_sync", REPO / "verl/utils/gate_a_weight_sync.py"
)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
sampled_tensor_digest = module.sampled_tensor_digest


@unittest.skipIf(torch is None, "torch is not installed in this CPU check environment")
class WeightSyncProjectionTests(unittest.TestCase):
    def test_projection_matches_effective_bfloat16_rollout_weights(self):
        name = "model.layers.0.input_layernorm.weight"
        initial = torch.tensor([1.0, 1.0], dtype=torch.float32)
        actor_after_small_update = torch.tensor(
            [1.0 + 1e-6, 1.0 - 1e-6], dtype=torch.float32
        )
        rollout = actor_after_small_update.to(torch.bfloat16)

        master_before = sampled_tensor_digest({name: initial}, [name], 2)
        master_after = sampled_tensor_digest({name: actor_after_small_update}, [name], 2)
        projected = sampled_tensor_digest(
            {name: actor_after_small_update}, [name], 2, project_to={name: rollout}
        )
        rollout_digest = sampled_tensor_digest({name: rollout}, [name], 2)

        self.assertNotEqual(master_before, master_after)
        self.assertEqual(projected, rollout_digest)
        self.assertNotEqual(master_after, rollout_digest)

    def test_projection_rejects_shape_mismatch(self):
        name = "model.norm.weight"
        with self.assertRaisesRegex(ValueError, "different shape"):
            sampled_tensor_digest(
                {name: torch.ones(3, dtype=torch.float32)},
                [name],
                3,
                project_to={name: torch.ones(2, dtype=torch.bfloat16)},
            )


if __name__ == "__main__":
    unittest.main()
