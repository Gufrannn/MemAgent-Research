#!/usr/bin/env python3
"""Read-only reconstruction of RWWPO actual-loss evidence."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def canonical_sha(record):
    payload = dict(record)
    payload.pop("record_sha256", None)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def audit(paths, require_method=True):
    rows, seen = [], set()
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                row = json.loads(line)
                if row.get("schema_version") != "rwwpo-actual-loss-v1":
                    raise ValueError(f"bad schema at {path}:{line_no}")
                if row.get("mode") not in ("rwwpo_method", "original_collection"):
                    raise ValueError(f"bad mode at {path}:{line_no}")
                if canonical_sha(row) != row.get("record_sha256"):
                    raise ValueError(f"record hash mismatch at {path}:{line_no}")
                identity = (row["attempt_id"], row["global_step"], row["rank"], row["epoch"], row["minibatch"])
                if identity in seen:
                    raise ValueError(f"duplicate optimizer identity: {identity}")
                seen.add(identity)
                shapes = [len(row[key]) for key in ("old_log_prob", "current_log_prob",
                          "response_mask", "writer_mask", "answer_mask", "advantages")]
                if len(set(shapes)) != 1 or shapes[0] != len(row["sample_index"]):
                    raise ValueError("row/tensor alignment failure")
                denominator = 0
                for response, writer, answer in zip(row["response_mask"], row["writer_mask"], row["answer_mask"]):
                    if not (len(response) == len(writer) == len(answer)):
                        raise ValueError("token shape failure")
                    for r, w, a in zip(response, writer, answer):
                        if bool(r) != (bool(w) ^ bool(a)):
                            raise ValueError("role mask closure failure")
                        denominator += int(bool(r))
                if denominator != row["denominator"]:
                    raise ValueError("denominator mismatch")
                for stat in row["prefix_stats"]:
                    expected = 1.0 / (1.0 + stat["chi2"])
                    if not math.isclose(stat["ess_fraction"], expected, rel_tol=1e-9, abs_tol=1e-12):
                        raise ValueError("ESS/chi-square identity failure")
                rows.append(row)
    if not rows:
        raise ValueError("missing actual-loss rows")
    active = any(any(abs(c-o) > 1e-10 for old, cur in zip(row["old_log_prob"], row["current_log_prob"])
                     for o, c in zip(old, cur)) for row in rows)
    if require_method and not active:
        raise ValueError("RWWPO_METHOD_INACTIVE")
    return {"status": "PASS", "decision": "RWWPO_ACTUAL_LOSS_LEDGER_PASS",
            "record_count": len(rows), "method_active": active,
            "modes": sorted({row["mode"] for row in rows}),
            "min_prefix_ess": min(s["ess_fraction"] for r in rows for s in r["prefix_stats"])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ledgers", nargs="+")
    parser.add_argument("--allow-behavior-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit(args.ledgers, not args.allow_behavior_only), sort_keys=True))


if __name__ == "__main__":
    main()
