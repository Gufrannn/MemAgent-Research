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
        previous = "0" * 64
        with Path(path).open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                row = json.loads(line)
                if row.get("schema_version") not in ("rwwpo-actual-loss-v1", "rwwpo-actual-loss-v2"):
                    raise ValueError(f"bad schema at {path}:{line_no}")
                if row.get("mode") not in ("rwwpo_method", "original_collection"):
                    raise ValueError(f"bad mode at {path}:{line_no}")
                if canonical_sha(row) != row.get("record_sha256"):
                    raise ValueError(f"record hash mismatch at {path}:{line_no}")
                if row.get("schema_version")=="rwwpo-actual-loss-v2":
                    if row.get("previous_record_sha256") != previous:
                        raise ValueError(f"append-only hash chain mismatch at {path}:{line_no}")
                    previous=row["record_sha256"]
                identity = (row["attempt_id"], row["global_step"], row["rank"], row["epoch"], row["minibatch"])
                if identity in seen:
                    raise ValueError(f"duplicate optimizer identity: {identity}")
                seen.add(identity)
                shapes = [len(row[key]) for key in ("old_log_prob", "current_log_prob",
                          "proposed_post_log_prob", "response_mask", "writer_mask", "answer_mask", "advantages")]
                if len(set(shapes)) != 1 or shapes[0] != len(row["sample_index"]):
                    raise ValueError("row/tensor alignment failure")
                if row["schema_version"] == "rwwpo-actual-loss-v2":
                    if len(row.get("example_identity_hash",[]))!=shapes[0] or len(row.get("trajectory_identity_hash",[]))!=shapes[0]:
                        raise ValueError("stable identity alignment failure")
                    for sid in set(row["sample_index"]):
                        idx=[i for i,value in enumerate(row["sample_index"]) if value==sid]
                        if len({row["example_identity_hash"][i] for i in idx})!=1 or len({row["trajectory_identity_hash"][i] for i in idx})!=1:
                            raise ValueError("stable trajectory identity drift")
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
                if row["schema_version"] == "rwwpo-actual-loss-v2":
                    if row.get("objective_variant") not in ("whole_prefix", "original_tokenwise"):
                        raise ValueError("objective variant identity drift")
                    if row.get("controller_variant") not in ("hard_rollback", "feasible_backtracking"):
                        raise ValueError("controller variant identity drift")
                    frozen=[1.0,0.5,0.25,0.125,0.0625,0.03125]
                    if row.get("controller_variant")=="feasible_backtracking" and row.get("alpha_grid")!=frozen:
                        raise ValueError("alpha grid drift")
                    alpha=float(row.get("alpha_committed",-1))
                    if alpha not in (0.0,*frozen): raise ValueError("invalid committed alpha")
                    if bool(row.get("accepted_nonzero")) != (alpha>0 and not bool(row.get("proposal_zero"))):
                        raise ValueError("zero proposal or accepted decision mismatch")
                    if bool(row.get("accepted")) != bool(row.get("accepted_nonzero")):
                        raise ValueError("legacy/new accepted decision mismatch")
                    required={"model","optimizer","scheduler","scaler","rng"}
                    if set(row.get("pre_digests",{}))!=required or set(row.get("commit_digests",{}))!=required:
                        raise ValueError("transaction digest closure failure")
                    if alpha==0:
                        for key in required:
                            if row["pre_digests"][key] != row["commit_digests"][key]:
                                raise ValueError("rollback digest mismatch")
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
                post_actual=[]
                post_logprob_field = "committed_log_prob" if row["schema_version"] == "rwwpo-actual-loss-v2" else "proposed_post_log_prob"
                for sid in sorted(set(row["sample_index"])):
                    indices=[i for i,value in enumerate(row["sample_index"]) if value==sid and any(bool(x) for x in row["writer_mask"][i])]
                    indices.sort(key=lambda i:row["trajectory_turn"][i]); running=0.0; tokens=0
                    for index in indices:
                        active=[j for j,value in enumerate(row["writer_mask"][index]) if bool(value)]
                        running += sum(float(row[post_logprob_field][index][j])-float(row["old_log_prob"][index][j]) for j in active); tokens += len(active)
                        post_actual.append({"turn":int(row["trajectory_turn"][index]),"sample_index":int(sid),"log_ratio":running,"prefix_token_count":tokens})
                post_declared=sorted(row["post_prefix_rows"],key=lambda x:(x["sample_index"],x["turn"])); post_actual.sort(key=lambda x:(x["sample_index"],x["turn"]))
                if len(post_declared)!=len(post_actual) or any(d["turn"]!=v["turn"] or d["sample_index"]!=v["sample_index"] or d["prefix_token_count"]!=v["prefix_token_count"] or not math.isclose(d["log_ratio"],v["log_ratio"],rel_tol=1e-9,abs_tol=1e-10) for d,v in zip(post_declared,post_actual)):
                    raise ValueError("post-step prefix rows do not reconstruct")
                if row["schema_version"] == "rwwpo-actual-loss-v2":
                    order=row["alpha_test_order"]
                    evidence=row["trial_evidence"]
                    if order != [item.get("alpha") for item in evidence]:
                        raise ValueError("trial test order/evidence mismatch")
                    if not order or order[0] != 1.0 or any(a not in row["alpha_grid"] for a in order):
                        raise ValueError("trial alpha evidence malformed")
                    if row["controller_variant"]=="feasible_backtracking" and order != row["alpha_grid"][:len(order)]:
                        raise ValueError("trial order is not descending grid prefix")
                    declared_alpha=float(row["alpha_committed"])
                    largest=next((float(item["alpha"]) for item in evidence if item.get("feasible")),0.0)
                    expected_alpha=0.0 if row["proposal_zero"] else largest
                    if declared_alpha != expected_alpha:
                        raise ValueError("committed alpha is not largest tested feasible")
                    for trial in evidence:
                        if len(trial.get("log_prob",[])) != len(row["old_log_prob"]):
                            raise ValueError("trial logprob row alignment failure")
                        trial_actual=[]
                        for sid in sorted(set(row["sample_index"])):
                            indices=[i for i,value in enumerate(row["sample_index"])
                                     if value==sid and any(bool(x) for x in row["writer_mask"][i])]
                            indices.sort(key=lambda i:row["trajectory_turn"][i]); running=0.0; tokens=0
                            for index in indices:
                                active=[j for j,value in enumerate(row["writer_mask"][index]) if bool(value)]
                                running += sum(float(trial["log_prob"][index][j])-float(row["old_log_prob"][index][j]) for j in active)
                                tokens += len(active)
                                trial_actual.append({"turn":int(row["trajectory_turn"][index]),
                                    "sample_index":int(sid),"log_ratio":running,"prefix_token_count":tokens})
                        trial_declared=sorted(trial["prefix_rows"],key=lambda x:(x["sample_index"],x["turn"]))
                        trial_actual.sort(key=lambda x:(x["sample_index"],x["turn"]))
                        if len(trial_declared)!=len(trial_actual) or any(
                            d["turn"]!=v["turn"] or d["sample_index"]!=v["sample_index"] or
                            d["prefix_token_count"]!=v["prefix_token_count"] or
                            not math.isclose(d["log_ratio"],v["log_ratio"],rel_tol=1e-9,abs_tol=1e-10)
                            for d,v in zip(trial_declared,trial_actual)):
                            raise ValueError("trial prefix rows do not reconstruct from actual logprobs")
                    if evidence[0]["log_prob"] != row["proposed_post_log_prob"]:
                        raise ValueError("full proposal logprob does not bind alpha=1 trial")
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
        def reconstruct(field):
            combined=[item for row in group for item in row[field]]; result=[]
            for turn in sorted({item["turn"] for item in combined}):
                values=[item["log_ratio"] for item in combined if item["turn"]==turn]
                peak=max(values); raw=[math.exp(value-peak) for value in values]; total=sum(raw)
                weights=[value/total for value in raw]; chi2=len(values)*sum(value*value for value in weights)-1
                result.append((turn,1/(1+chi2),chi2,max(abs(value) for value in values)))
            return result
        for rows_field, stats_field in (("prefix_rows","prefix_stats"),("post_prefix_rows","post_prefix_stats")):
          expected=reconstruct(rows_field)
          for row in group:
            declared=row[stats_field]
            if len(declared)!=len(expected) or any(item["turn"]!=value[0] or not math.isclose(item["ess_fraction"],value[1],rel_tol=1e-9,abs_tol=1e-10) or not math.isclose(item["chi2"],value[2],rel_tol=1e-9,abs_tol=1e-10) or not math.isclose(item["max_abs_log_ratio"],value[3],rel_tol=1e-9,abs_tol=1e-10) for item,value in zip(declared,expected)):
                raise ValueError(f"global {stats_field} do not reconstruct for {key}")
        q_values={float(row["q_min"]) for row in group}; caps={float(row["writer_log_ratio_cap"]) for row in group}
        decisions={bool(row["accepted"]) for row in group}
        if len(q_values)!=1 or len(caps)!=1 or len(decisions)!=1: raise ValueError(f"distributed decision disagreement for {key}")
        if group[0]["schema_version"] == "rwwpo-actual-loss-v2":
            orders={tuple(float(x) for x in row["alpha_test_order"]) for row in group}
            if len(orders)!=1: raise ValueError(f"distributed trial order disagreement for {key}")
            for trial_alpha in next(iter(orders)):
                combined=[]
                declared_feasible=set()
                for row in group:
                    trial=[item for item in row["trial_evidence"] if float(item["alpha"])==trial_alpha]
                    if len(trial)!=1: raise ValueError(f"trial evidence missing for {key}")
                    combined.extend(trial[0]["prefix_rows"]); declared_feasible.add(bool(trial[0]["feasible"]))
                if len(declared_feasible)!=1: raise ValueError(f"distributed trial feasibility drift for {key}")
                actual_feasible=True
                for turn in sorted({item["turn"] for item in combined}):
                    values=[item["log_ratio"] for item in combined if item["turn"]==turn]
                    peak=max(values); weights=[math.exp(v-peak) for v in values]; total=sum(weights)
                    weights=[v/total for v in weights]; chi2=len(values)*sum(v*v for v in weights)-1
                    actual_feasible &= 1/(1+chi2)>=next(iter(q_values)) and max(abs(v) for v in values)<=next(iter(caps))
                if next(iter(declared_feasible)) != actual_feasible:
                    raise ValueError(f"forged trial feasibility for {key} alpha={trial_alpha}")
            alpha_values={float(row["alpha_committed"]) for row in group}
            if len(alpha_values)!=1: raise ValueError(f"distributed alpha disagreement for {key}")
            alpha=next(iter(alpha_values))
            expected_accept=alpha>0 and not any(bool(row["proposal_zero"]) for row in group)
            if next(iter(decisions)) != expected_accept:
                raise ValueError(f"accepted decision is not certified by committed alpha for {key}")
            if alpha>0:
                selected=[]
                for row in group:
                    match=[trial for trial in row["trial_evidence"] if float(trial["alpha"])==alpha]
                    if len(match)!=1: raise ValueError(f"committed trial missing for {key}")
                    selected.extend(match[0]["prefix_rows"])
                values_by_turn={turn:[item["log_ratio"] for item in selected if item["turn"]==turn]
                                for turn in sorted({item["turn"] for item in selected})}
                feasible=True
                for values in values_by_turn.values():
                    peak=max(values); weights=[math.exp(v-peak) for v in values]; total=sum(weights)
                    weights=[v/total for v in weights]; chi2=len(values)*sum(v*v for v in weights)-1
                    feasible &= 1/(1+chi2)>=next(iter(q_values)) and max(abs(v) for v in values)<=next(iter(caps))
                if not feasible: raise ValueError(f"committed alpha is globally infeasible for {key}")
        else:
            expected_accept=all(s[1]>=next(iter(q_values)) and s[3]<=next(iter(caps)) for s in reconstruct("post_prefix_rows"))
            if next(iter(decisions)) != expected_accept: raise ValueError(f"accepted decision is not certified by post statistics for {key}")
        pre_pass=all(s[1]>=next(iter(q_values)) and s[3]<=next(iter(caps)) for s in reconstruct("prefix_rows"))
        if any(bool(row["constraint_pass"])!=pre_pass for row in group): raise ValueError(f"pre-step constraint decision mismatch for {key}")
    # On-policy behavior-point closure requires current == old before the
    # optimizer step; using that equality's violation as an activity signal
    # incorrectly rejects a healthy on-policy run.  Method activity is instead
    # certified by a non-zero proposed post-step movement that survived the
    # distributed trust-region acceptance decision.
    behavior_point_max_delta = max(abs(c-o) for row in rows
        for old,cur in zip(row["old_log_prob"],row["current_log_prob"])
        for o,c in zip(old,cur))
    active = any(bool(group[0]["accepted"]) and
        any(abs(p-c) > 1e-10 for row in group
            for post,cur in zip(row["proposed_post_log_prob"],row["current_log_prob"])
            for p,c in zip(post,cur)) for group in groups.values())
    if require_method and not active:
        raise ValueError("RWWPO_METHOD_INACTIVE")
    step_summaries={}
    for step in sorted({int(r["global_step"]) for r in rows}):
        selected=[r for r in rows if int(r["global_step"])==step]
        step_groups=[g for key,g in groups.items() if int(key[1])==step]
        step_summaries[str(step)]={"accepted_fraction":sum(bool(g[0]["accepted"]) for g in step_groups)/len(step_groups),
            "alpha_committed":[float(g[0].get("alpha_committed",1.0 if g[0]["accepted"] else 0.0)) for g in step_groups],
            "accepted_nonzero_count":sum(bool(g[0].get("accepted_nonzero",g[0]["accepted"])) for g in step_groups),
            "max_proposed_update":max(abs(p-c) for r in selected for post,cur in zip(r["proposed_post_log_prob"],r["current_log_prob"]) for p,c in zip(post,cur))}
    v2_groups=[group for group in groups.values() if group[0]["schema_version"]=="rwwpo-actual-loss-v2"]
    nonzero_alphas=[float(group[0]["alpha_committed"]) for group in v2_groups
                    if bool(group[0]["accepted_nonzero"])]
    return {"status": "PASS", "decision": "RWWPO_ACTUAL_LOSS_LEDGER_PASS",
            "record_count": len(rows), "method_active": active,
            "behavior_point_max_delta": behavior_point_max_delta,
            "modes": sorted({row["mode"] for row in rows}),
            "objective_variants": sorted({row.get("objective_variant","legacy") for row in rows}),
            "controller_variants": sorted({row.get("controller_variant","legacy") for row in rows}),
            "nonzero_commit_count": len(nonzero_alphas),
            "committed_nonzero_alphas": nonzero_alphas,
            "min_prefix_ess": min(s["ess_fraction"] for r in rows for s in r["prefix_stats"]),
            "min_post_prefix_ess": min(s["ess_fraction"] for r in rows for s in r["post_prefix_stats"]),
            "accepted_fraction": sum(bool(next(iter(g))["accepted"]) for g in groups.values())/len(groups),
            "max_proposed_update": max(abs(p-c) for r in rows for post,cur in zip(r["proposed_post_log_prob"],r["current_log_prob"]) for p,c in zip(post,cur)),
            "steps":step_summaries}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ledgers", nargs="+")
    parser.add_argument("--allow-behavior-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit(args.ledgers, not args.allow_behavior_only), sort_keys=True))


if __name__ == "__main__":
    main()
