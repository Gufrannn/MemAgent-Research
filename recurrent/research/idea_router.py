"""Single real training-path hook for the evidence-gated 7B arms."""
from __future__ import annotations
import json, os
from pathlib import Path
import torch
from .idea_admissibility import require_arm
from .ncr_certified_routing import FORBIDDEN_OUTCOME_FIELDS

SCORER_FIELDS = {"ncr_certified_routing": "ncr_secondary_score", "generic_qa_aux": "generic_qa_score",
 "generic_frozen_judge_tournament": "frozen_judge_score", "information_matched_raw_judge": "raw_judge_score",
 "uniform_tie_rescue": "uniform_direction"}

def _manifest_rows(path: str) -> dict[str, dict]:
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    keys = [str(row.get("trajectory_id", "")) for row in records]
    if any(not key for key in keys) or len(keys) != len(set(keys)): raise ValueError("NO_METHOD: manifest trajectory_id must be unique")
    return dict(zip(keys, records))

def _append_audit(records: list[dict]) -> None:
    path = os.environ.get("IDEA_REWARD_AUDIT")
    if not path: raise ValueError("NO_METHOD: IDEA_REWARD_AUDIT is required")
    output = Path(path); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a") as stream:
        for record in records: stream.write(json.dumps(record, sort_keys=True) + "\n")

def apply_idea_arm(*, trajectory_qa_advantage: torch.Tensor, batch, reward_batch,
                   reward_tensor: torch.Tensor, sample_index: torch.Tensor) -> torch.Tensor:
    arm = os.environ.get("IDEA_ARM", "qa_only_original")
    require_arm(arm, os.environ.get("IDEA_EVIDENCE_LEDGER"))
    response_mask = batch.batch["response_mask"]
    qa_rows = trajectory_qa_advantage[sample_index].unsqueeze(-1).expand_as(response_mask) * response_mask
    if arm == "qa_only_original": return qa_rows
    manifest_path = os.environ.get("IDEA_REWARD_MANIFEST", "")
    if not manifest_path: raise ValueError("NO_METHOD: IDEA_REWARD_MANIFEST is required")
    manifest = _manifest_rows(manifest_path)
    trajectory_ids = [str(x) for x in reward_batch.non_tensor_batch.get("trajectory_id", [])]
    if len(trajectory_ids) != len(trajectory_qa_advantage): raise ValueError("NO_METHOD: identity/reward tuple length mismatch")
    rows = []
    for trajectory_id in trajectory_ids:
        if trajectory_id not in manifest: raise ValueError(f"NO_METHOD: missing manifest row {trajectory_id}")
        row = manifest[trajectory_id]
        if arm != "ncr_certified_routing" and set(row) & FORBIDDEN_OUTCOME_FIELDS:
            raise ValueError("NO_METHOD: generic baseline contains BOT/NOOP labels")
        rows.append(row)
    scorer_field = SCORER_FIELDS[arm]
    secondary = torch.tensor([float(row[scorer_field]) for row in rows], device=trajectory_qa_advantage.device)
    eligible = torch.tensor([bool(row["eligible"]) for row in rows], device=secondary.device)
    exact_correct = torch.tensor([bool(row["exact_correct"]) for row in rows], device=secondary.device)
    uids = [str(x) for x in reward_batch.non_tensor_batch["uid"]]; qa_reward = reward_tensor.sum(-1)
    bonus = torch.zeros_like(secondary); routed = torch.zeros(len(rows), dtype=torch.bool, device=secondary.device)
    for uid in dict.fromkeys(uids):
        idx = torch.tensor([value == uid for value in uids], device=secondary.device); group_reward = qa_reward[idx]
        if not bool(torch.all(group_reward == group_reward[0])): continue
        if bool(torch.all(exact_correct[idx])): raise ValueError("NO_METHOD: all-exact-correct tie group")
        if not bool(torch.all(eligible[idx])): continue
        scores = secondary[idx]; bonus[idx] = float(os.environ.get("IDEA_LAMBDA", "0")) * (scores - scores.mean()); routed[idx] = True
    final_rows = batch.batch["final_mask"].bool(); writer_mask = response_mask * (~final_rows).unsqueeze(-1)
    row_bonus = bonus[sample_index].unsqueeze(-1).expand_as(qa_rows) * writer_mask; advantages = qa_rows + row_bonus
    if not torch.equal(advantages[~routed[sample_index]], qa_rows[~routed[sample_index]]): raise AssertionError("non-tie advantage changed")
    if torch.any(row_bonus[final_rows] != 0): raise AssertionError("final-answer secondary is nonzero")
    audit = [{"uid": uids[i], "trajectory_id": trajectory_ids[i], "trajectory_seed": int(reward_batch.non_tensor_batch["trajectory_seed"][i]),
      "sample_index": i, "replica_role": row["replica_role"], "qa_reward": float(qa_reward[i]), "secondary_score": float(secondary[i]),
      "secondary_bonus": float(bonus[i]), "normalization": "same_uid_mean_center", "lambda": float(os.environ.get("IDEA_LAMBDA", "0")),
      "stratum": row["stratum"], "eligible": bool(eligible[i]), "writer_mask_tokens": int(writer_mask[sample_index == i].sum()),
      "manifest_hash": os.environ.get("IDEA_MANIFEST_HASH"), "readout_hash": os.environ.get("NCR_FROZEN_READOUT_HASH"),
      "seed_schedule": "independent"} for i, row in enumerate(rows)]
    _append_audit(audit); return advantages
