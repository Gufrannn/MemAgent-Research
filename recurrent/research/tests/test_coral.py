import unittest
from recurrent.research.coral import exposure_counts, phase_for_step, validate_config

try:
    import torch
    from recurrent.research.coral import role_covered_order, role_masks
    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False


class CoralPureContractTests(unittest.TestCase):
    def test_count_sketch_bases_have_distinct_collision_and_sign_maps(self):
        from recurrent.research.coral_e1 import (
            SKETCH_SPEC, sketch_bucket_and_sign, sketch_coordinate_offset,
        )
        mask = (1 << 63) - 1
        for ordinal in (0, 7, 63, 10_000):
            for rank in range(2):
                for basis in range(4):
                    offset = sketch_coordinate_offset(ordinal, rank, basis)
                    expected = (
                        ordinal * SKETCH_SPEC["parameter_multiplier"]
                        + rank * SKETCH_SPEC["rank_multiplier"]
                        + basis * SKETCH_SPEC["basis_multiplier"]
                    ) & mask
                    self.assertEqual(offset, expected)
                    self.assertGreaterEqual(offset, 0)
                    self.assertLessEqual(offset, mask)
        coordinates = range(10000)
        mappings = [
            [sketch_bucket_and_sign(i, i % 17, i % 2, basis)
             for i in coordinates]
            for basis in range(SKETCH_SPEC["independent_bases"])
        ]
        for mapping in mappings:
            self.assertEqual(len({bucket for bucket, _ in mapping}), 256)
        for other in mappings[1:]:
            same_bucket = sum(
                left[0] == right[0] for left, right in zip(mappings[0], other)
            ) / len(mappings[0])
            same_sign = sum(
                left[1] == right[1] for left, right in zip(mappings[0], other)
            ) / len(mappings[0])
            bucket_deltas = {
                (right[0] - left[0]) % 256
                for left, right in zip(mappings[0], other)
            }
            self.assertLess(same_bucket, 0.02)
            self.assertGreater(same_sign, 0.45)
            self.assertLess(same_sign, 0.55)
            self.assertGreater(len(bucket_deltas), 200)

    def test_terminal_contrast_seeds_are_common_and_turn_independent(self):
        from recurrent.research.trajectory_seeding import (
            derive_coral_terminal_contrast_seeds,
        )
        source = [11, 22, 33]
        first = derive_coral_terminal_contrast_seeds(source)
        self.assertEqual(first, derive_coral_terminal_contrast_seeds(source))
        self.assertEqual(len(set(first)), 3)
        self.assertNotEqual(first, source)

    def test_schedule_starts_at_update_one_and_resumes_from_step(self):
        self.assertEqual([phase_for_step(i) for i in range(1, 7)],
                         ["memory_writer", "terminal_answer"] * 3)
        self.assertEqual(exposure_counts(25), {"memory_writer": 13, "terminal_answer": 12})
        with self.assertRaisesRegex(ValueError, "CORAL_NO_GO"):
            phase_for_step(0)

    def test_config_is_exact(self):
        good={"enabled":True,"active_from_update":1,
              "schedule":"odd_writer_even_terminal_answer_v2",
              "role_partition":"nonfinal_memory_writer_vs_final_answer",
              "require_recurrent":True,"require_grpo":True,"require_gate_a_sync":True}
        self.assertEqual(validate_config(good),good)
        bad=dict(good);bad["role_partition"]="all_turn_boundaries"
        with self.assertRaisesRegex(ValueError,"CORAL_NO_GO"):
            validate_config(bad)


@unittest.skipUnless(TORCH_AVAILABLE, "torch is provided by the H20 workspace runtime")
class CoralTensorContractTests(unittest.TestCase):
    def test_masks_partition_writer_and_final_tokens(self):
        mask = torch.tensor([[1,1,0],[1,0,0],[1,1,1]])
        final = torch.tensor([False, True, False])
        writer, answer = role_masks(mask, final, "memory_writer")
        answer_active, writer_inactive = role_masks(mask, final, "terminal_answer")
        self.assertTrue(torch.equal(writer, writer_inactive))
        self.assertTrue(torch.equal(answer, answer_active))
        self.assertTrue(torch.equal(writer + answer, mask))

    def test_malformed_or_single_role_batches_fail(self):
        with self.assertRaisesRegex(ValueError, "CORAL_NO_GO"):
            role_masks(torch.ones(2,3), torch.tensor([True, True]), "writer")
        with self.assertRaisesRegex(ValueError, "CORAL_NO_GO"):
            role_masks(torch.ones(2,3), torch.tensor([0,1]), "writer")

    def test_role_covered_order_covers_each_rank_and_preserves_padding(self):
        final = torch.tensor([False, False, False, False, True, True, False, False])
        valid = torch.tensor([True, True, True, True, True, True, False, False])
        order = role_covered_order(final, 2, valid)
        self.assertEqual(sorted(order.tolist()), list(range(8)))
        for shard in order.reshape(2, 4):
            real = shard[valid[shard]]
            self.assertTrue(bool(final[real].any()))
            self.assertTrue(bool((~final[real]).any()))

if __name__ == "__main__": unittest.main()
