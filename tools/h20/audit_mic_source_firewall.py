#!/usr/bin/env python3
"""Static taint audit for every trainer-side MIC critic-state constructor."""

import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TRAINER = REPO / "verl/trainer/ppo/ray_trainer.py"
ALLOWED = {
    "stable_example_id", "stable_root_id", "trajectory_id", "turn_index",
    "question", "visible_chunks", "materialized_memory",
    "materialized_memory_history", "is_prewrite",
}
FORBIDDEN_NAMES = {
    "reward", "reward_tensor", "outcome", "gold", "ground_truth",
    "generated_answer", "future_chunk", "answer_text",
}


def main() -> int:
    tree = ast.parse(TRAINER.read_text(encoding="utf-8"), filename=str(TRAINER))
    constructors = []
    failures = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) \
                or node.func.attr != "append" or not isinstance(node.func.value, ast.Name) \
                or node.func.value.id != "states" or len(node.args) != 1 \
                or not isinstance(node.args[0], ast.Dict):
            continue
        keys = {key.value for key in node.args[0].keys if isinstance(key, ast.Constant)}
        constructors.append((node.lineno, keys))
        if keys != ALLOWED:
            failures.append(f"line {node.lineno} critic keys {sorted(keys)} != {sorted(ALLOWED)}")
        names = {child.id for child in ast.walk(node.args[0]) if isinstance(child, ast.Name)}
        leaked = names & FORBIDDEN_NAMES
        if leaked:
            failures.append(f"line {node.lineno} critic state references forbidden names {sorted(leaked)}")
    if len(constructors) != 2:
        failures.append(f"expected exactly two critic state constructors, found {len(constructors)}")
    if failures:
        print("MIC_SOURCE_FIREWALL_FAIL:" + "; ".join(failures), file=sys.stderr)
        return 2
    print("MIC_SOURCE_FIREWALL_PASS:" + ",".join(str(line) for line, _ in constructors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
