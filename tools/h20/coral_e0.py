#!/usr/bin/env python3
"""Enumerated shared-policy recurrent-MDP counterexample for CORAL.

There is one shared scalar parameter. A writer samples a one-bit materialized
memory and the same policy, under a role-conditioned feature, samples a terminal
answer. Reward is one exactly when the answer equals memory. The group-size-2
GRPO estimator is enumerated over all 16 ordered rollout pairs with the frozen
Original mean-centered (``grpo_use_adv=False``) outcome advantages; no Monte
Carlo sample is used.
"""
from __future__ import annotations

import argparse
import json
import sys
from decimal import Decimal, getcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from recurrent.research.cosi import canonical_sha256

getcontext().prec = 60
D = Decimal

# role-conditioned logits of one shared theta:
# writer: 10 theta - 4; answer|M=0: -9 theta - 3; answer|M=1: -4 theta - 3.
PARAMETERS = {"writer": (10, -4), "answer_m0": (-9, -3), "answer_m1": (-4, -3)}


def sigmoid(value: Decimal) -> Decimal:
    return D(1) / (D(1) + (-value).exp())


def enumerate_policy(theta: Decimal):
    aw, bw = PARAMETERS["writer"]
    p = sigmoid(D(aw) * theta + D(bw))
    q = {}
    for memory, name in ((0, "answer_m0"), (1, "answer_m1")):
        slope, bias = PARAMETERS[name]
        q[memory] = sigmoid(D(slope) * theta + D(bias))
    events = []
    for memory in (0, 1):
        pm = p if memory else D(1) - p
        for answer in (0, 1):
            pa = q[memory] if answer else D(1) - q[memory]
            reward = D(int(answer == memory))
            events.append({
                "probability": pm * pa,
                "reward": reward,
                "writer_score": D(aw) * (D(memory) - p),
                "answer_score": D(PARAMETERS[f"answer_m{memory}"][0])
                * (D(answer) - q[memory]),
            })
    expected_return = sum((x["probability"] * x["reward"] for x in events), D(0))
    writer_gradient = D(0)
    answer_gradient = D(0)
    for left in events:
        for right in events:
            if left["reward"] == right["reward"]:
                continue  # zero-variance group -> zero normalized advantages
            # The accepted Original has grpo_use_adv=False.  For a two-sample
            # group, subtracting the group mean gives +/- (R_l - R_r) / 2.
            left_advantage = (left["reward"] - right["reward"]) / D(2)
            right_advantage = -left_advantage
            pair_probability = left["probability"] * right["probability"]
            writer_gradient += pair_probability * (
                left_advantage * left["writer_score"]
                + right_advantage * right["writer_score"]
            ) / D(2)
            answer_gradient += pair_probability * (
                left_advantage * left["answer_score"]
                + right_advantage * right["answer_score"]
            ) / D(2)
    return expected_return, writer_gradient, answer_gradient


def text(value: Decimal) -> str:
    return format(value, ".24f")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    theta_old, learning_rate = D("0.6"), D("4.7")
    old_return, writer_gradient, stale_answer_gradient = enumerate_policy(theta_old)
    # The real actor aggregates Original over both active tokens. Each role
    # gradient above is already a one-role token mean, so the full-token
    # denominator contributes exactly 1/2 to both simultaneous and masked
    # updates. CORAL keeps that Original denominator in code.
    full_token_scale = D("0.5")
    theta_simultaneous = theta_old + learning_rate * full_token_scale * (
        writer_gradient + stale_answer_gradient
    )
    simultaneous_return = enumerate_policy(theta_simultaneous)[0]

    theta_writer = theta_old + learning_rate * full_token_scale * writer_gradient
    writer_refreshed_return, _, refreshed_answer_gradient = enumerate_policy(theta_writer)
    theta_coral = theta_writer + learning_rate * full_token_scale * refreshed_answer_gradient
    coral_return = enumerate_policy(theta_coral)[0]

    simultaneous_reversal = simultaneous_return < old_return - D("0.03")
    coral_improvement = coral_return > old_return + D("0.03")
    # A materialized-memory refresh has changed the terminal-answer update
    # direction.  The direction is model-dependent; the identifiable event is
    # the strict sign reversal, not one arbitrarily chosen orientation.
    response_identified = stale_answer_gradient * refreshed_answer_gradient < 0
    passed = simultaneous_reversal and coral_improvement and response_identified
    report = {
        "schema": "memagent.coral.e0.v3",
        "status": "PASS" if passed else "FAIL",
        "decision": "CORAL_E0_PASS" if passed else "CORAL_E0_NO_GO",
        "environment": "one-bit materialized memory followed by terminal answer",
        "reward": "1[terminal_answer == materialized_memory]",
        "shared_parameter": True,
        "role_conditioned_logits": {key: list(value) for key, value in PARAMETERS.items()},
        "estimator": "exact enumeration of group-size-2 mean-centered GRPO outcome advantages (grpo_use_adv=false)",
        "ordered_group_pairs": 16,
        "loss_aggregation": "token-mean over the same two-token Original denominator",
        "per_role_full_token_scale": text(full_token_scale),
        "learning_rate": text(learning_rate),
        "old": {
            "theta": text(theta_old),
            "return": text(old_return),
            "writer_gradient": text(writer_gradient),
            "stale_terminal_answer_gradient": text(stale_answer_gradient),
        },
        "simultaneous_masked_grpo": {
            "theta": text(theta_simultaneous),
            "return": text(simultaneous_return),
        },
        "coral": {
            "writer_proposal_theta": text(theta_writer),
            "return_after_writer_proposal": text(writer_refreshed_return),
            "refreshed_terminal_answer_gradient": text(refreshed_answer_gradient),
            "final_theta": text(theta_coral),
            "return": text(coral_return),
        },
        "simultaneous_reversal": simultaneous_reversal,
        "coral_improvement": coral_improvement,
        "occupancy_response_sign_flip": response_identified,
    }
    report["report_sha256"] = canonical_sha256(report)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
