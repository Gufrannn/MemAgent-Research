import hashlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from recurrent.research import mic_v2_e1 as e1
from recurrent.research.mic_v2 import sha256_file, sha256_json
from tools.h20.mic_v2_e1_pipeline import (
    _coverage_cells, _exact_materialized_memory_token_count, _materialize_sources,
    _open_holdout_once, _project_split,
    _root_cluster_diagnostic_intervals, _validate_cross_turn_filtration,
    _verify_inherited_lock_authority,
)
from tools.h20 import run_qwen25_7b_mic_v2_e1_collect as e1_collector
from tools.h20 import run_qwen25_7b_mic_v2_e1_features as e1_features


def root(index: int) -> str:
    return hashlib.sha256(f"root-{index}".encode()).hexdigest()


class MicV2E1CoreTest(unittest.TestCase):
    def test_gpu_collector_rechecks_git_and_model_before_loading(self):
        repo = Path(__file__).resolve().parents[2]
        expected_commit = "a" * 40
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve()
            with mock.patch.object(
                e1_collector, "_git", side_effect=[expected_commit, "dirty-helper"],
            ), self.assertRaisesRegex(RuntimeError, "collector Git authority"):
                e1_collector.collect(repo, expected_commit, output, "dev-r1", "produce")

            certificates = output / "certificates"
            certificates.mkdir()
            manifest_path = repo / e1_collector.MANIFEST_REL
            relative_self = str(Path(e1_collector.__file__).resolve().relative_to(repo))
            code_authority = {relative_self: sha256_file(Path(e1_collector.__file__))}
            p0 = {
                "schema": "memagent.mic.v2.e1-p0", "split": "e1_dev",
                "git_commit": expected_commit, "run_id": "dev-r1",
                "output_root": str(output), "gpu_pair": [0, 1],
                "manifest_sha256": sha256_file(manifest_path),
                "code_sha256": code_authority,
                "model_files": [{"path": "expected-model-file"}],
            }
            p0["p0_sha256"] = sha256_json(p0)
            (certificates / "p0.json").write_text(json.dumps(p0))
            with mock.patch.object(
                e1_collector, "_git", side_effect=[expected_commit, ""],
            ), mock.patch.object(
                e1_collector, "_verify_inherited_lock_authority", return_value=[0, 1],
            ), mock.patch.object(
                e1_collector, "_strict_vllm_environment", return_value={},
            ), mock.patch.object(
                e1_collector, "_gpu_identity", return_value=[],
            ), mock.patch.object(
                e1_collector, "_source_firewall", return_value=code_authority,
            ), mock.patch.object(
                e1_collector, "_verify_model", return_value=[{"path": "changed-model-file"}],
            ), mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1"}), \
                    self.assertRaisesRegex(RuntimeError, "code/model authority"):
                e1_collector.collect(repo, expected_commit, output, "dev-r1", "produce")

    def test_gpu_feature_entry_rechecks_git_and_model_before_loading(self):
        repo = Path(__file__).resolve().parents[2]
        expected_commit = "b" * 40
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary).resolve()
            with mock.patch.object(
                e1_features, "_git", side_effect=[expected_commit, "dirty-helper"],
            ), self.assertRaisesRegex(RuntimeError, "feature Git authority"):
                e1_features.extract(
                    repo, expected_commit, output, "dev-r1", "e1_dev", "produce",
                )

            certificates = output / "certificates"
            certificates.mkdir()
            manifest_path = repo / e1_features.MANIFEST_REL
            relative_self = str(Path(e1_features.__file__).resolve().relative_to(repo))
            code_authority = {relative_self: sha256_file(Path(e1_features.__file__))}
            p0 = {
                "schema": "memagent.mic.v2.e1-p0", "split": "e1_dev",
                "git_commit": expected_commit, "run_id": "dev-r1",
                "output_root": str(output), "gpu_pair": [0, 1],
                "manifest_sha256": sha256_file(manifest_path),
                "code_sha256": code_authority,
                "model_files": [{"path": "expected-model-file"}],
            }
            p0["p0_sha256"] = sha256_json(p0)
            (certificates / "p0.json").write_text(json.dumps(p0))
            with mock.patch.object(
                e1_features, "_git", side_effect=[expected_commit, ""],
            ), mock.patch.object(
                e1_features, "_source_firewall", return_value=code_authority,
            ), mock.patch.object(
                e1_features, "_verify_model", return_value=[{"path": "changed-model-file"}],
            ), self.assertRaisesRegex(RuntimeError, "code/model authority"):
                e1_features.extract(
                    repo, expected_commit, output, "dev-r1", "e1_dev", "produce",
                )

    def test_candidate_grid_is_exact_and_finite(self):
        specs = e1.candidate_specs()
        self.assertEqual(len(specs), 48)
        self.assertEqual({spec.regularization for spec in specs}, set(e1.LAMBDAS))
        self.assertEqual({spec.head for spec in specs}, set(e1.HEADS))
        self.assertEqual({spec.representation for spec in specs},
                         {item[0] for item in e1.REPRESENTATIONS})

    def test_stable_fold_is_global_and_set_invariant(self):
        roots = [root(index) for index in range(50)]
        expected = {item: e1.stable_selection_fold(item) for item in roots}
        for item in reversed(roots[3:41]):
            self.assertEqual(e1.stable_selection_fold(item), expected[item])
        with self.assertRaisesRegex(ValueError, "content root"):
            e1.stable_selection_fold("not-a-root")

    def test_root_trajectory_turn_weights_are_hierarchical(self):
        roots = [root(0)] * 3 + [root(1)] * 4
        trajectories = ["a", "a", "b", "c", "c", "d", "d"]
        turns = [1, 2, 1, 1, 2, 1, 2]
        weights = e1.root_trajectory_turn_weights(roots, trajectories, turns)
        self.assertAlmostEqual(float(weights.sum()), 1.0)
        self.assertAlmostEqual(float(weights[:3].sum()), 0.5)
        self.assertAlmostEqual(float(weights[3:].sum()), 0.5)
        self.assertAlmostEqual(float(weights[:2].sum()), 0.25)
        self.assertAlmostEqual(float(weights[2]), 0.25)

    def test_signed_hash_and_turn_features_are_frozen(self):
        components = {
            "question": "Who wrote it?", "arrived_history": "One. Two!",
            "current_memory": "A memory",
        }
        first = e1.signed_text_hash(components, turn=2)
        second = e1.signed_text_hash(components, turn=2)
        self.assertEqual(first.shape, (4096,))
        np.testing.assert_array_equal(first, second)
        self.assertEqual(
            hashlib.sha256(first.tobytes()).hexdigest(),
            "3b58b6809aa76d2a031cae40338e0259c94d2431890099836ab6eedefd052157",
        )
        with self.assertRaisesRegex(ValueError, "schema/order"):
            e1.signed_text_hash({**components, "content_root_id": root(1)}, turn=2)
        self.assertFalse(np.array_equal(first, e1.signed_text_hash(components, turn=3)))
        hidden = [0.25, -0.5, 1.0]
        projected = e1.rademacher_projection(hidden, 128, turn=2)
        self.assertEqual(projected.shape, (128,))
        np.testing.assert_array_equal(
            projected, e1.rademacher_projection(hidden, 128, turn=2),
        )
        self.assertFalse(np.array_equal(
            projected, e1.rademacher_projection(hidden, 128, turn=3),
        ))
        interactions = e1.actor_hidden_interactions(
            [1.0, 0.0, 0.5], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [0.0, 0.5, 1.0],
        )
        self.assertEqual(interactions.shape, (448,))
        self.assertTrue(np.isfinite(interactions).all())
        np.testing.assert_array_equal(
            e1.turn_length_features(
                turn=2, arrived_chunk_count=2, prior_active_turn_count=1,
                arrived_context_token_count=999, current_memory_token_count=17,
            ),
            np.asarray([2, 2, 1, 999, 17], dtype=np.float64),
        )

    def test_turn_length_uses_exact_materialized_token_receipt_not_decoded_text(self):
        row = {
            "pre_state": {"current_memory": "decoded text may re-tokenize differently"},
            "post_state": {"current_memory": "same visible text is not the authority"},
            "pre_materialized_memory_token_count": 2,
            "pre_materialized_memory_token_sha256": sha256_json([17, 23]),
            "post_materialized_memory_token_count": 3,
            "post_materialized_memory_token_sha256": sha256_json([101, 102, 103]),
        }
        self.assertEqual(_exact_materialized_memory_token_count(row, "pre"), 2)
        self.assertEqual(_exact_materialized_memory_token_count(row, "post"), 3)
        row["post_materialized_memory_token_sha256"] = "not-a-digest"
        with self.assertRaisesRegex(RuntimeError, "exact materialized-memory"):
            _exact_materialized_memory_token_count(row, "post")

    def test_turn_zero_post_features_and_stage_mask_are_explicit(self):
        components = e1.text_components_from_state({
            "question": "q", "arrived_chunks": [], "current_memory": "",
        }, no_memory_text="No previous memory")
        self.assertEqual(components["arrived_history"], "")
        np.testing.assert_array_equal(
            e1.turn_length_features(
                turn=0, arrived_chunk_count=0, prior_active_turn_count=0,
                arrived_context_token_count=0, current_memory_token_count=0,
            ),
            np.zeros(5, dtype=np.float64),
        )
        interaction = e1.actor_hidden_interactions(
            [1.0, 0.0], [], [0.0, 1.0],
        )
        self.assertEqual(interaction.shape, (448,))
        self.assertTrue(np.isfinite(e1.rademacher_projection(
            interaction, 128, turn=0,
        )).all())

        roots, trajectories, turns, target, pre, post = [], [], [], [], [], []
        for index in range(16):
            item = root(index)
            for turn in (0, 1):
                roots.append(item)
                trajectories.append(f"trajectory-{index}")
                turns.append(turn)
                target.append(index / 15)
                pre.append([0.0] if turn == 0 else [index / 15])
                post.append([index / 15])
        mask = {
            "pre": np.asarray([turn == 1 for turn in turns]),
            "post": np.ones(len(turns), dtype=bool),
        }
        spec = e1.CandidateSpec("toy", 1, 0, "bounded_ridge", 0.1)
        prediction = e1.cross_fitted_predictions(
            spec, root_ids=roots, trajectory_ids=trajectories, turns=turns,
            target=target,
            stage_features={"pre": np.asarray(pre), "post": np.asarray(post)},
            stage_masks=mask,
        )
        self.assertTrue(np.isnan(prediction["pre"][~mask["pre"]]).all())
        self.assertTrue(np.isfinite(prediction["pre"][mask["pre"]]).all())
        self.assertTrue(np.isfinite(prediction["post"]).all())

    def test_cross_turn_filtration_rejects_individually_valid_but_stale_history(self):
        common = {
            "question": "q", "content_root_id": root(0),
            "trajectory_id": root(1),
            "public_metadata": {"chunk_schedule_id": root(2)},
        }
        previous = {
            **common, "arrived_chunks": ["c1"],
            "materialized_memory_history": ["M1"], "current_memory": "M1",
        }
        valid = {
            **common, "arrived_chunks": ["c1", "c2"],
            "materialized_memory_history": ["M1"], "current_memory": "M1",
            "public_metadata": {
                "chunk_schedule_id": root(2), "prior_active_turn_count": 1,
            },
        }
        _validate_cross_turn_filtration(valid, previous, 2, root(2))
        stale = {**valid, "materialized_memory_history": ["DIFFERENT"],
                 "current_memory": "DIFFERENT"}
        with self.assertRaisesRegex(RuntimeError, "cross-turn filtration"):
            _validate_cross_turn_filtration(stale, previous, 2, root(2))

    def test_symmetric_heads_fit_continuous_return(self):
        x = np.asarray([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
        y = np.asarray([0.05, 0.2, 0.5, 0.8, 0.95])
        w = np.ones(5) / 5
        for head in e1.HEADS:
            spec = e1.CandidateSpec("toy", 1, 0, head, 1e-2)
            fitted = e1.fit_head(spec, x, y, w)
            prediction = fitted.predict(x)
            self.assertTrue(np.isfinite(prediction).all())
            self.assertTrue(((0 <= prediction) & (prediction <= 1)).all())
            self.assertLess(e1.weighted_mse(y, prediction, w), 0.04)

    def test_fractional_logistic_extreme_logit_gradient_matches_objective(self):
        features = np.asarray([[3.0, -2.0], [-4.0, 5.0], [1.0, 7.0]])
        target = np.asarray([0.1, 0.8, 0.35])
        weights = np.asarray([0.2, 0.3, 0.5])
        parameters = np.asarray([60.0, 25.0, -18.0])
        loss, analytic = e1._fractional_logistic_objective(
            parameters, features, target, weights, 0.01,
        )
        numeric = np.empty_like(parameters)
        epsilon = 1e-5
        for index in range(parameters.size):
            upper, lower = parameters.copy(), parameters.copy()
            upper[index] += epsilon
            lower[index] -= epsilon
            upper_loss = e1._fractional_logistic_objective(
                upper, features, target, weights, 0.01,
            )[0]
            lower_loss = e1._fractional_logistic_objective(
                lower, features, target, weights, 0.01,
            )[0]
            numeric[index] = (upper_loss - lower_loss) / (2.0 * epsilon)
        self.assertTrue(np.isfinite(loss))
        np.testing.assert_allclose(analytic, numeric, atol=2e-8, rtol=2e-8)

    def test_dual_ridge_matches_primal_and_head_receipt_round_trips(self):
        generator = np.random.default_rng(20260825)
        x = generator.normal(size=(12, 128))
        y = np.linspace(0.05, 0.95, 12)
        w = np.arange(1, 13, dtype=np.float64)
        spec = e1.CandidateSpec("actor_hidden_rademacher_128", 128, 2, "bounded_ridge", 0.1)
        fitted = e1.fit_head(spec, x, y, w)
        standardizer = e1.fit_standardizer(x, w)
        normalized_weight = w / w.sum()
        z = standardizer.apply(x)
        y_mean = float(normalized_weight @ y)
        expected = np.linalg.solve(
            z.T @ (normalized_weight[:, None] * z) + 0.1 * np.eye(128),
            z.T @ (normalized_weight * (y - y_mean)),
        )
        np.testing.assert_allclose(fitted.coefficients, expected, atol=1e-10, rtol=1e-10)
        restored = e1.FittedHead.from_receipt(fitted.receipt())
        np.testing.assert_allclose(restored.predict(x), fitted.predict(x), atol=0, rtol=0)

    def test_four_fold_selection_uses_shared_spec_and_prefers_signal(self):
        roots = []
        index = 0
        while len(roots) < 32:
            candidate = root(index)
            roots.append(candidate)
            index += 1
        self.assertEqual(set(map(e1.stable_selection_fold, roots)), {0, 1, 2, 3})
        trajectories = [f"trajectory-{index}" for index in range(32)]
        turns = [1] * 32
        signal = np.linspace(-2.0, 2.0, 32)
        target = 1.0 / (1.0 + np.exp(-signal))
        noise = np.zeros((32, 1))
        useful = signal[:, None]
        specs = (
            e1.CandidateSpec("noise", 1, 0, "bounded_ridge", 1e-2),
            e1.CandidateSpec("useful", 1, 1, "bounded_ridge", 1e-2),
        )
        with mock.patch.object(e1, "candidate_specs", return_value=specs), \
             mock.patch.object(e1, "REPRESENTATIONS", (("noise", 1, 0), ("useful", 1, 1))):
            report = e1.select_specification(
                root_ids=roots, trajectory_ids=trajectories, turns=turns,
                target=target,
                features={
                    "noise": {"pre": noise, "post": noise},
                    "useful": {"pre": useful, "post": useful},
                },
            )
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["selected"]["specification"]["representation"], "useful")

    def test_coverage_is_occupied_cell_fraction_and_all_diagnostics_are_clustered(self):
        by_fold = {fold: [] for fold in range(4)}
        index = 0
        while any(len(values) < 25 for values in by_fold.values()):
            candidate = root(index)
            fold = e1.stable_selection_fold(candidate)
            if len(by_fold[fold]) < 25:
                by_fold[fold].append(candidate)
            index += 1
        root_ids, trajectories, turns = [], [], []
        for fold in range(4):
            for candidate in by_fold[fold]:
                root_ids.append(candidate)
                trajectories.append(f"trajectory-{candidate}")
                turns.append(1)
            root_ids.append(by_fold[fold][0])
            trajectories.append(f"trajectory-{by_fold[fold][0]}")
            turns.append(2)
        occupied, fraction = _coverage_cells(root_ids, turns, 8, 24)
        self.assertEqual(len(occupied), 8)
        self.assertEqual(fraction, 0.5)
        # A row-weighted implementation would incorrectly report 100 / 104.
        self.assertNotAlmostEqual(fraction, 100 / 104)
        coordinate = np.linspace(0.05, 0.95, len(root_ids))
        target = 0.1 + 0.8 * coordinate
        pre = 0.15 + 0.7 * coordinate
        post = np.clip(pre + 0.03 * np.sin(np.arange(len(pre))), 0, 1)
        intervals = _root_cluster_diagnostic_intervals(
            target=target, pre=pre, post=post, credit=post - pre,
            root_ids=root_ids, trajectory_ids=trajectories, turns=turns,
            occupied_cells=occupied, minimum_roots_per_cell=24,
            replicates=200, seed=20260825,
        )
        self.assertEqual(intervals["coverage"]["eligible_cell_fraction"], 0.5)
        self.assertEqual(len(intervals["coverage"]["cells"]), 8)
        self.assertEqual(len(intervals["calibration"]["pre"]["slope_two_sided_95"]), 2)
        self.assertEqual(
            len(intervals["writer_credit"]["weighted_variance_two_sided_95"]), 2,
        )

    def test_e1_source_projection_separates_gpu_text_from_cpu_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root_path = Path(temporary).resolve()
            rows, frozen = [], []
            for index in range(128):
                question, context, ground_truth = f"question {index}", f"context {index}", [f"answer {index}"]
                rows.append({
                    "prompt": [{"role": "user", "content": question}], "context": context,
                    "extra_info": {"index": index, "num_docs": 2, "question": question},
                    "reward_model": {"ground_truth": ground_truth, "style": "rule"},
                })
                frozen.append({
                    "source_position": index, "semantic_dataset_index": index,
                    "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
                    "context_sha256": hashlib.sha256(context.encode()).hexdigest(),
                    "content_root_id": root(index),
                    "ground_truth_sha256": sha256_json(ground_truth),
                })
            parquet = root_path / "train.parquet"
            pq.write_table(pa.Table.from_pylist(rows), parquet)
            resolved = {"splits": {"e1_dev": {
                "content_root_count": 128, "rows": frozen,
            }}}
            split = _project_split(resolved, "e1_dev")
            split["projection_sha256"] = sha256_json(split)
            gpu, outcomes = _materialize_sources(
                {"source": {"path": str(parquet), "sha256": sha256_file(parquet)}},
                split, root_path / "attempt",
            )
            gpu_rows = [json.loads(line) for line in Path(gpu["path"]).read_text().splitlines()]
            outcome_rows = [json.loads(line) for line in Path(outcomes["path"]).read_text().splitlines()]
            self.assertEqual(len(gpu_rows), 128)
            self.assertEqual(len(outcome_rows), 128)
            self.assertNotIn("ground_truth", gpu_rows[0])
            self.assertNotIn("question", outcome_rows[0])
            self.assertEqual(outcome_rows[0]["ground_truth"], ["answer 0"])

    def test_holdout_opening_is_exactly_once_and_selection_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            dev = Path(temporary) / "dev"
            holdout = Path(temporary) / "holdout"
            certificates = dev / "certificates"
            certificates.mkdir(parents=True)
            p0 = {
                "schema": "p0", "run_id": "dev-r1", "git_commit": "a" * 40,
                "output_root": str(dev), "split": "e1_dev",
            }
            p0["p0_sha256"] = sha256_json(p0)
            (certificates / "p0.json").write_text(json.dumps(p0))
            selection = {
                "schema": "memagent.mic.v2.e1-selection", "status": "PASS",
                "decision": "MIC_V2_E1_SPECIFICATION_SELECTED",
                "git_commit": "a" * 40, "run_id": "dev-r1",
                "p0_sha256": p0["p0_sha256"], "holdout_opened": False,
            }
            selection["selection_sha256"] = sha256_json(selection)
            (certificates / "e1_dev_selection.json").write_text(json.dumps(selection))
            holdout_certificates = holdout / "certificates"
            holdout_certificates.mkdir(parents=True)
            gpu_receipt = {
                "schema": "memagent.mic.v2.e1-dev-opening-gpu-reverification",
                "status": "PASS", "git_commit": "a" * 40,
                "dev_output_root": str(dev), "holdout_output_root": str(holdout),
                "holdout_run_id": "holdout-r1",
            }
            gpu_receipt["reverification_sha256"] = sha256_json(gpu_receipt)
            (holdout_certificates / "e1_dev_opening_gpu_reverification.json").write_text(
                json.dumps(gpu_receipt)
            )
            replay_target = "MEMAGENT_MIC_V2_E1_REPLAY_TARGET_RUN_ID"

            def replay_selection(*arguments, **keywords):
                self.assertEqual(os.environ.get(replay_target), "dev-r1")
                return selection

            with mock.patch.dict(os.environ, {replay_target: "outer-target"}), mock.patch(
                "tools.h20.mic_v2_e1_pipeline.select_dev",
                side_effect=replay_selection,
            ) as replay:
                receipt = _open_holdout_once(
                    Path(temporary), dev, holdout, "a" * 40, "holdout-r1",
                )
                replay.assert_called_once_with(
                    Path(temporary), "a" * 40, dev, "dev-r1", verify_existing=True,
                )
                self.assertEqual(os.environ.get(replay_target), "outer-target")
            self.assertEqual(receipt["status"], "OPENED_ONCE")
            with mock.patch(
                "tools.h20.mic_v2_e1_pipeline.select_dev", return_value=selection,
            ), self.assertRaises(FileExistsError):
                _open_holdout_once(
                    Path(temporary), dev, holdout, "a" * 40, "holdout-r1",
                )
            other = Path(temporary) / "other" / "certificates"
            other.mkdir(parents=True)
            other_p0 = {**p0, "output_root": str(other.parent)}
            other_p0.pop("p0_sha256")
            other_p0["p0_sha256"] = sha256_json(other_p0)
            (other / "p0.json").write_text(json.dumps(other_p0))
            with mock.patch.dict(os.environ, {replay_target: "outer-target"}), mock.patch(
                "tools.h20.mic_v2_e1_pipeline.select_dev",
                side_effect=RuntimeError("full dev replay differs"),
            ), self.assertRaisesRegex(RuntimeError, "full dev replay"):
                _open_holdout_once(
                    Path(temporary), other.parent, holdout, "a" * 40, "holdout-r2",
                )
            self.assertNotEqual(os.environ.get(replay_target), "dev-r1")

    def test_real_shells_are_fresh_only_and_replay_is_unconditional(self):
        repo = Path(__file__).resolve().parents[2]
        dev = (repo / "scripts/h20/run_qwen25_7b_mic_v2_e1_dev.sh").read_text()
        holdout = (repo / "scripts/h20/run_qwen25_7b_mic_v2_e1_holdout.sh").read_text()
        for source in (dev, holdout):
            self.assertNotIn("RESUME", source)
            self.assertIn("for mode in produce replay", source)
            self.assertNotIn('if [[ ! -e "$ROOT/', source)
        self.assertIn("verify-dev", dev)
        self.assertIn("verify-holdout", holdout)
        self.assertIn("MIC_V2_E1_HOLDOUT_PASS", holdout)
        self.assertNotIn("EVALUATION_FINISHED", holdout)
        self.assertIn("e1_holdout_opening.json", inspect.getsource(_open_holdout_once))
        invoked = subprocess.run(
            ["bash", str(repo / "scripts/h20/run_qwen25_7b_mic_v2_e1_dev.sh")],
            env={"PATH": "/usr/bin:/bin"}, text=True, capture_output=True,
        )
        self.assertNotEqual(invoked.returncode, 0)
        self.assertIn("missing MEMAGENT_MIC_V2_WORK_ROOT", invoked.stderr)

    def test_dev_shell_real_entry_rejects_wrong_commit_dirty_and_existing_attempt(self):
        entry = Path(__file__).resolve().parents[2] / "scripts/h20/run_qwen25_7b_mic_v2_e1_dev.sh"
        with tempfile.TemporaryDirectory() as temporary:
            root_path = Path(temporary).resolve()
            repo = root_path / "repo"
            repo.mkdir()
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.email", "e1@test"], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.name", "E1 Test"], check=True)
            (repo / "tracked").write_text("frozen")
            subprocess.run(["git", "-C", str(repo), "add", "tracked"], check=True)
            subprocess.run(["git", "-C", str(repo), "commit", "-qm", "frozen"], check=True)
            commit = subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True,
            ).strip()
            work = root_path / "work"
            python = work / ".venv/bin/python"
            python.parent.mkdir(parents=True)
            python.write_text("#!/bin/sh\nexit 0\n")
            python.chmod(0o755)
            base = {
                "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                "MEMAGENT_MIC_V2_WORK_ROOT": str(work),
                "MEMAGENT_MIC_V2_REPO_DIR": str(repo),
                "MEMAGENT_MIC_V2_EXPECTED_COMMIT": "a" * 40,
                "MEMAGENT_MIC_V2_E1_DEV_RUN_ID": "e1-dev-entry-test",
                "MEMAGENT_MIC_V2_GPU_PAIR": "0,1",
            }
            wrong = subprocess.run(
                ["bash", str(entry)], env=base, text=True, capture_output=True,
            )
            self.assertEqual(wrong.returncode, 46)
            dirty_env = {**base, "MEMAGENT_MIC_V2_EXPECTED_COMMIT": commit}
            (repo / "untracked").write_text("dirty")
            dirty = subprocess.run(
                ["bash", str(entry)], env=dirty_env, text=True, capture_output=True,
            )
            self.assertEqual(dirty.returncode, 47)
            (repo / "untracked").unlink()
            attempt = work / "logs/mic_v2_e1/e1-dev-entry-test"
            attempt.mkdir(parents=True)
            existing = subprocess.run(
                ["bash", str(entry)], env=dirty_env, text=True, capture_output=True,
            )
            self.assertEqual(existing.returncode, 48)
            optimized = subprocess.run(
                ["bash", str(entry)], env={**dirty_env, "PYTHONOPTIMIZE": "1"},
                text=True, capture_output=True,
            )
            self.assertEqual(optimized.returncode, 45)

    def test_holdout_shell_propagates_preflight_and_final_replay_failures(self):
        entry = Path(__file__).resolve().parents[2] / "scripts/h20/run_qwen25_7b_mic_v2_e1_holdout.sh"

        def invoke(failure_stage: str):
            temporary = tempfile.TemporaryDirectory()
            self.addCleanup(temporary.cleanup)
            root_path = Path(temporary.name).resolve()
            repo = root_path / "repo"
            repo.mkdir()
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.email", "e1@test"], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.name", "E1 Test"], check=True)
            (repo / "tracked").write_text("frozen")
            subprocess.run(["git", "-C", str(repo), "add", "tracked"], check=True)
            subprocess.run(["git", "-C", str(repo), "commit", "-qm", "frozen"], check=True)
            commit = subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True,
            ).strip()
            work = root_path / "work"
            fake_bin = root_path / "bin"
            fake_bin.mkdir()
            for name, source in (
                ("flock", "#!/bin/sh\nexit 0\n"),
                ("nvidia-smi", "#!/bin/sh\nexit 0\n"),
            ):
                path = fake_bin / name
                path.write_text(source)
                path.chmod(0o755)
            python = work / ".venv/bin/python"
            python.parent.mkdir(parents=True)
            python.write_text(
                "#!/bin/sh\n"
                "case \"$*\" in\n"
                "  *preflight-holdout*)\n"
                "    if [ \"$FAKE_FAILURE_STAGE\" = preflight ]; then exit 91; fi\n"
                "    mkdir -p \"$MEMAGENT_MIC_V2_WORK_ROOT/logs/mic_v2_e1/"
                "$MEMAGENT_MIC_V2_E1_HOLDOUT_RUN_ID/certificates\";;\n"
                "  *verify-holdout*)\n"
                "    if [ \"$FAKE_FAILURE_STAGE\" = verify ]; then exit 92; fi;;\n"
                "esac\n"
                "exit 0\n"
            )
            python.chmod(0o755)
            dev = work / "logs/mic_v2_e1/dev-r1/certificates"
            dev.mkdir(parents=True)
            (dev / "e1_dev_selection.json").write_text("{}")
            environment = {
                "PATH": f"{fake_bin}:/usr/bin:/bin",
                "FAKE_FAILURE_STAGE": failure_stage,
                "MEMAGENT_MIC_V2_WORK_ROOT": str(work),
                "MEMAGENT_MIC_V2_REPO_DIR": str(repo),
                "MEMAGENT_MIC_V2_EXPECTED_COMMIT": commit,
                "MEMAGENT_MIC_V2_E1_DEV_RUN_ID": "dev-r1",
                "MEMAGENT_MIC_V2_E1_HOLDOUT_RUN_ID": f"holdout-{failure_stage}",
                "MEMAGENT_MIC_V2_GPU_PAIR": "0,1",
            }
            completed = subprocess.run(
                ["bash", str(entry)], env=environment, text=True, capture_output=True,
            )
            output = work / "logs/mic_v2_e1" / f"holdout-{failure_stage}"
            return completed, output, work / "logs/mic_v2_e1/dev-r1"

        preflight, preflight_root, dev_root = invoke("preflight")
        self.assertEqual(preflight.returncode, 91)
        self.assertFalse((preflight_root / "certificates/p0.json").exists())
        self.assertFalse((dev_root / "certificates/e1_holdout_opening.json").exists())
        self.assertFalse((preflight_root / "authorities/cpu_outcomes.jsonl").exists())

        verification, verification_root, _dev_root = invoke("verify")
        self.assertEqual(verification.returncode, 92)
        self.assertFalse(
            (verification_root / "certificates/e1_holdout_final_verification.json").exists()
        )

    def test_holdout_shell_rejects_lock_conflict_and_occupied_gpu(self):
        entry = Path(__file__).resolve().parents[2] / "scripts/h20/run_qwen25_7b_mic_v2_e1_holdout.sh"
        with tempfile.TemporaryDirectory() as temporary:
            root_path = Path(temporary).resolve()
            repo = root_path / "repo"
            repo.mkdir()
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.email", "e1@test"], check=True)
            subprocess.run(["git", "-C", str(repo), "config", "user.name", "E1 Test"], check=True)
            (repo / "tracked").write_text("frozen")
            subprocess.run(["git", "-C", str(repo), "add", "tracked"], check=True)
            subprocess.run(["git", "-C", str(repo), "commit", "-qm", "frozen"], check=True)
            commit = subprocess.check_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True,
            ).strip()
            work = root_path / "work"
            python = work / ".venv/bin/python"
            python.parent.mkdir(parents=True)
            python.write_text("#!/bin/sh\nexit 0\n")
            python.chmod(0o755)
            dev = work / "logs/mic_v2_e1/dev-r1/certificates"
            dev.mkdir(parents=True)
            (dev / "e1_dev_selection.json").write_text("{}")
            fake_bin = root_path / "bin"
            fake_bin.mkdir()
            nvidia = fake_bin / "nvidia-smi"
            nvidia.write_text("#!/bin/sh\necho 424242\n")
            nvidia.chmod(0o755)
            flock = fake_bin / "flock"
            flock.write_text("#!/bin/sh\nexit 0\n")
            flock.chmod(0o755)
            environment = {
                "PATH": f"{fake_bin}:/usr/bin:/bin",
                "MEMAGENT_MIC_V2_WORK_ROOT": str(work),
                "MEMAGENT_MIC_V2_REPO_DIR": str(repo),
                "MEMAGENT_MIC_V2_EXPECTED_COMMIT": commit,
                "MEMAGENT_MIC_V2_E1_DEV_RUN_ID": "dev-r1",
                "MEMAGENT_MIC_V2_E1_HOLDOUT_RUN_ID": "holdout-occupied",
                "MEMAGENT_MIC_V2_GPU_PAIR": "0,1",
            }
            occupied = subprocess.run(
                ["bash", str(entry)], env=environment, text=True, capture_output=True,
            )
            self.assertEqual(occupied.returncode, 72)
            nvidia.write_text("#!/bin/sh\nexit 0\n")
            flock.write_text("#!/bin/sh\nexit 1\n")
            environment["MEMAGENT_MIC_V2_E1_HOLDOUT_RUN_ID"] = "holdout-lock"
            locked = subprocess.run(
                ["bash", str(entry)], env=environment, text=True, capture_output=True,
            )
            self.assertEqual(locked.returncode, 69)

    def test_forged_official_marker_without_inherited_lock_fds_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary).resolve()
            locks = work / "locks"
            locks.mkdir()
            run_id = "forged-marker"
            paths = [
                locks / f"memagent_mic_v2_e1_{run_id}.lock",
                locks / "memagent_h20_gpu_0.lock",
                locks / "memagent_h20_gpu_1.lock",
            ]
            for path in paths:
                path.touch()
            environment = {
                "MEMAGENT_MIC_V2_E1_OFFICIAL_ENTRY": "locked-shell-v1",
                "MEMAGENT_MIC_V2_E1_LOCK_RUN_ID": run_id,
                "MEMAGENT_MIC_V2_E1_LOCK_WORK_ROOT": str(work),
                "MEMAGENT_MIC_V2_E1_LOCK_GPU_PAIR": "0,1",
                "MEMAGENT_MIC_V2_E1_LOCK_FDS": "7,8,9",
                **{
                    f"MEMAGENT_MIC_V2_E1_LOCK_PATH_{fd}": str(path)
                    for fd, path in zip((7, 8, 9), paths)
                },
            }
            with mock.patch.dict(os.environ, environment, clear=True), \
                    self.assertRaisesRegex(RuntimeError, "lock proof|wrong file"):
                _verify_inherited_lock_authority(run_id)

    def test_inherited_lock_proof_uses_real_kernel_flock(self):
        import fcntl

        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary).resolve()
            locks = work / "locks"
            locks.mkdir()
            run_id = "kernel-lock-proof"
            paths = [
                locks / f"memagent_mic_v2_e1_{run_id}.lock",
                locks / "memagent_h20_gpu_0.lock",
                locks / "memagent_h20_gpu_1.lock",
            ]
            for path in paths:
                path.touch()
            environment = {
                **os.environ,
                "MEMAGENT_MIC_V2_E1_OFFICIAL_ENTRY": "locked-shell-v1",
                "MEMAGENT_MIC_V2_E1_LOCK_RUN_ID": run_id,
                "MEMAGENT_MIC_V2_E1_LOCK_WORK_ROOT": str(work),
                "MEMAGENT_MIC_V2_E1_LOCK_GPU_PAIR": "0,1",
                "MEMAGENT_MIC_V2_E1_LOCK_FDS": "7,8,9",
                **{
                    f"MEMAGENT_MIC_V2_E1_LOCK_PATH_{fd}": str(path)
                    for fd, path in zip((7, 8, 9), paths)
                },
            }
            script = (
                "import fcntl, os\n"
                "from pathlib import Path\n"
                "paths=[Path(os.environ[f'MEMAGENT_MIC_V2_E1_LOCK_PATH_{fd}']) "
                "for fd in (7,8,9)]\n"
                "for fd,path in zip((7,8,9),paths):\n"
                "  opened=os.open(path,os.O_WRONLY)\n"
                "  if opened != fd:\n"
                "    os.dup2(opened,fd); os.close(opened)\n"
                "  if os.environ.get('LOCK_BEFORE') == '1': "
                "fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
                "from tools.h20.mic_v2_e1_pipeline import _verify_inherited_lock_authority\n"
                "print(_verify_inherited_lock_authority(os.environ['EVIDENCE_RUN_ID']))\n"
            )
            accepted = subprocess.run(
                [sys.executable, "-c", script],
                env={**environment, "LOCK_BEFORE": "1", "EVIDENCE_RUN_ID": run_id},
                text=True, capture_output=True,
            )
            self.assertEqual(accepted.returncode, 0, accepted.stderr)
            self.assertIn("[0, 1]", accepted.stdout)

            held = os.open(paths[0], os.O_WRONLY)
            try:
                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
                rejected = subprocess.run(
                    [sys.executable, "-c", script],
                    env={**environment, "LOCK_BEFORE": "0", "EVIDENCE_RUN_ID": run_id},
                    text=True, capture_output=True,
                )
                self.assertNotEqual(rejected.returncode, 0)
                self.assertIn("lock proof failed", rejected.stderr)
            finally:
                os.close(held)

            dev_replay = subprocess.run(
                [sys.executable, "-c", script],
                env={
                    **environment, "LOCK_BEFORE": "1",
                    "EVIDENCE_RUN_ID": "dev-evidence-run",
                    "MEMAGENT_MIC_V2_E1_REPLAY_TARGET_RUN_ID": "dev-evidence-run",
                },
                text=True, capture_output=True,
            )
            self.assertEqual(dev_replay.returncode, 0, dev_replay.stderr)


if __name__ == "__main__":
    unittest.main()
