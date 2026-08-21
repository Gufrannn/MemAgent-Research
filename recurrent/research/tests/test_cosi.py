import hashlib
import unittest

from recurrent.research.cosi import (
    SCHEMA, canonical_sha256, decide, derive_seed, root_contrasts,
    validate_four_cell_bundle,
)


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def bundle(scores=None, roots=("r0", "r1", "r2", "r3")):
    scores = scores or {"OO": 0.0, "NO": 0.5, "ON": 0.0, "NN": 0.25}
    old, new, base = _sha("old"), _sha("new"), 2026
    records = []
    for root in roots:
        for cell in ("OO", "NO", "ON", "NN"):
            writer, cont = cell
            candidate = _sha(root + writer)
            future = [derive_seed(base_seed=base, root_id=root, writer=writer,
                                  replica=0, phase="future", turn=i) for i in range(2)]
            records.append({
                "root_id": root, "replica": 0, "cell": cell,
                "writer_checkpoint": writer, "continuation_checkpoint": cont,
                "candidate_sha256": candidate, "candidate_token_ids_sha256": candidate,
                "writer_seed": derive_seed(base_seed=base, root_id=root, writer=writer,
                                             replica=0, phase="writer"),
                "future_seeds": future,
                "writer_weight_sha256": old if writer == "O" else new,
                "continuation_weight_sha256": old if cont == "O" else new,
                "score": scores[cell], "score_evidence_sha256": _sha(root + cell),
            })
    contract = {
        "old_weight_sha256": old, "new_weight_sha256": new, "base_seed": base,
        "reward_low": -1.0, "reward_high": 1.0,
        "candidate_sampling": "cache_once_no_resample",
        "future_seed_coupling": "common_within_root_writer_replica",
        "root_inventory_sha256": canonical_sha256(sorted(roots)),
        "git_commit": "a" * 40, "transport_manifest_sha256": _sha("manifest"),
    }
    unsigned = {"schema": SCHEMA, "contract": contract, "records": records}
    return {**unsigned, "bundle_sha256": canonical_sha256(unsigned)}


def resign(value):
    value["bundle_sha256"] = canonical_sha256({k: v for k, v in value.items() if k != "bundle_sha256"})


class CosiTests(unittest.TestCase):
    def test_exact_decomposition_and_decision(self):
        value = bundle(); rows = root_contrasts(value)
        self.assertEqual(rows[0]["writer_old"], .5)
        self.assertEqual(rows[0]["interaction"], -.25)
        self.assertEqual(rows[0]["closed"], .25)
        self.assertEqual(decide(value, alpha=.05, delta=3.0, attempt=0, alpha_schedule=[.05])["decision"], "ACCEPT")

    def test_adversarial_receipts_fail_closed(self):
        for mutation in ("missing", "candidate", "future", "weight", "seed", "duplicate", "inactive"):
            with self.subTest(mutation=mutation):
                value = bundle()
                if mutation == "missing": value["records"].pop()
                elif mutation == "candidate": value["records"][1]["candidate_sha256"] = _sha("fake")
                elif mutation == "future": value["records"][1]["future_seeds"][0] += 1
                elif mutation == "weight": value["records"][0]["writer_weight_sha256"] = _sha("fake")
                elif mutation == "seed": value["records"][0]["writer_seed"] += 1
                elif mutation == "duplicate": value["records"][1]["cell"] = "OO"
                elif mutation == "inactive": value["contract"]["new_weight_sha256"] = value["contract"]["old_weight_sha256"]
                resign(value)
                with self.assertRaisesRegex(ValueError, "COSI_NO_GO"):
                    validate_four_cell_bundle(value)

    def test_signature_tampering_rejected(self):
        value = bundle(); value["records"][0]["score"] = 1.0
        with self.assertRaisesRegex(ValueError, "authentication"):
            validate_four_cell_bundle(value)

    def test_root_clustering_not_cell_pseudoreplication(self):
        result = decide(bundle(roots=("only-root",)), alpha=.05, delta=0, attempt=0, alpha_schedule=[.05])
        self.assertEqual(result["root_count"], 1); self.assertEqual(result["decision"], "ROLLBACK")

if __name__ == "__main__":
    unittest.main()
