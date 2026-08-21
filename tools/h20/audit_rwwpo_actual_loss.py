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
                          "proposed_post_log_prob", "response_mask", "writer_mask", "answer_mask", "advantages")]
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
                recomputed=[]
                for sid in sorted(set(row["sample_index"])):
                    indices=[i for i,value in enumerate(row["sample_index"]) if value==sid and any(bool(x) for x in row["writer_mask"][i])]
                    indices.sort(key=lambda i:row["trajectory_turn"][i]); running=0.0; tokens=0
                    for index in indices:
                        active=[j for j,value in enumerate(row["writer_mask"][index]) if bool(value)]
                        advantages={round(float(row["advantages"][index][j]),12) for j in active}
                        if len(advantages)!=1: raise ValueError("writer advantage is not scalar within a write")
                        running += sum(float(row["current_log_prob"][index][j])-float(row["old_log_prob"][index][j]) for j in active)
                        tokens += len(active)
                        recomputed.append({"turn":int(row["trajectory_turn"][index]),"sample_index":int(sid),"log_ratio":running,"prefix_token_count":tokens})
                declared=sorted(row["prefix_rows"],key=lambda x:(x["sample_index"],x["turn"]))
                actual=sorted(recomputed,key=lambda x:(x["sample_index"],x["turn"]))
                if len(declared)!=len(actual) or any(d["turn"]!=v["turn"] or d["sample_index"]!=v["sample_index"] or d["prefix_token_count"]!=v["prefix_token_count"] or not math.isclose(d["log_ratio"],v["log_ratio"],rel_tol=1e-9,abs_tol=1e-10) for d,v in zip(declared,actual)):
                    raise ValueError("prefix rows do not reconstruct from actual-loss tensors")
                for stat in row["prefix_stats"]:
                    expected = 1.0 / (1.0 + stat["chi2"])
                    if not math.isclose(stat["ess_fraction"], expected, rel_tol=1e-9, abs_tol=1e-12):
                        raise ValueError("ESS/chi-square identity failure")
                rows.append(row)
    if not rows:
        raise ValueError("missing actual-loss rows")
    groups={}
    for row in rows:
        key=(row["attempt_id"],row["global_step"],row["epoch"],row["minibatch"])
        groups.setdefault(key,[]).append(row)
    for key,group in groups.items():
        combined=[item for row in group for item in row["prefix_rows"]]
        expected=[]
        for turn in sorted({item["turn"] for item in combined}):
            values=[item["log_ratio"] for item in combined if item["turn"]==turn]
            peak=max(values); raw=[math.exp(value-peak) for value in values]; total=sum(raw)
            weights=[value/total for value in raw]; chi2=len(values)*sum(value*value for value in weights)-1
            expected.append((turn,1/(1+chi2),chi2,max(abs(value) for value in values)))
        for row in group:
            declared=row["prefix_stats"]
            if len(declared)!=len(expected) or any(item["turn"]!=value[0] or not math.isclose(item["ess_fraction"],value[1],rel_tol=1e-9,abs_tol=1e-10) or not math.isclose(item["chi2"],value[2],rel_tol=1e-9,abs_tol=1e-10) or not math.isclose(item["max_abs_log_ratio"],value[3],rel_tol=1e-9,abs_tol=1e-10) for item,value in zip(declared,expected)):
                raise ValueError(f"global prefix statistics do not reconstruct for {key}")
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
