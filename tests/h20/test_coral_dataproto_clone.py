import copy
import unittest

try:
    import numpy as np
    import torch
    from tensordict import TensorDict
    from verl import DataProto
except (ImportError, OSError):
    np = None
    torch = None
    TensorDict = None
    DataProto = None


@unittest.skipIf(DataProto is None, "torch/tensordict runtime unavailable")
class CoralDataProtoCloneTests(unittest.TestCase):
    def test_zero_leaf_batch_clone_and_deepcopy(self):
        source = DataProto(
            batch=TensorDict({}, batch_size=(8,)),
            non_tensor_batch={
                "uid": np.asarray([f"root-{index}" for index in range(8)], dtype=object),
                "index": np.arange(8, dtype=np.int64),
            },
            meta_info={"nested": {"phase": "source"}},
        )

        cloned = source.clone()
        self.assertEqual(len(cloned), 8)
        self.assertEqual(list(cloned.batch.keys()), [])
        cloned.non_tensor_batch["uid"][0] = "mutated"
        cloned.meta_info["nested"]["phase"] = "mutated"
        self.assertEqual(source.non_tensor_batch["uid"][0], "root-0")
        self.assertEqual(source.meta_info["nested"]["phase"], "source")

        # The serialization guard prevents the original H20 torch.cat([])
        # failure even if an unrelated caller still uses generic deepcopy.
        serialized_clone = copy.deepcopy(source)
        self.assertEqual(len(serialized_clone), 8)
        self.assertEqual(list(serialized_clone.batch.keys()), [])
        serialized_clone.non_tensor_batch["uid"][0] = "serialized-mutation"
        serialized_clone.meta_info["nested"]["phase"] = "serialized-mutation"
        self.assertEqual(source.non_tensor_batch["uid"][0], "root-0")
        self.assertEqual(source.meta_info["nested"]["phase"], "source")

    def test_tensor_payload_is_not_aliased(self):
        source = DataProto(
            batch=TensorDict({"value": torch.arange(8)}, batch_size=(8,)),
            non_tensor_batch={},
            meta_info={},
        )
        cloned = source.clone()
        cloned.batch["value"][0] = -1
        self.assertEqual(int(source.batch["value"][0]), 0)


if __name__ == "__main__":
    unittest.main()
