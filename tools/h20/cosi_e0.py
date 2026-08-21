#!/usr/bin/env python3
"""Run the CPU-only COSI E0 gate and emit an authenticated report."""
from __future__ import annotations

import argparse
import json
import math
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recurrent.research.cosi import canonical_sha256


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    # A shared scalar theta controls writer P(B)=1/2+theta and continuation
    # success P(success|B)=4/5-20 theta^2; P(success|A)=1/5.  At theta=0 a
    # two-root GRPO group (B-success reward 1; A-failure reward 0, hence
    # centered standardized advantages +1/-1) has gradient dS/dtheta=1 at
    # theta=0. SGD-ascent with registered learning rate 1/10 therefore yields
    # theta'=1/10. Its tokenwise clipped surrogate is positive, while the
    # exact full return falls from 1/2 to 11/25.  All quantities are rational.
    old_theta, new_theta = Fraction(0), Fraction(1, 10)
    old_return = Fraction(1, 2) * Fraction(4, 5) + Fraction(1, 2) * Fraction(1, 5)
    new_return = Fraction(3, 5) * Fraction(3, 5) + Fraction(2, 5) * Fraction(1, 5)
    # Positive sample: writer ratio 6/5, answer-success ratio 3/4.
    # Negative sample: writer ratio 4/5, unchanged answer-failure ratio 1.
    clip_epsilon = Fraction(1, 5)
    sampled_tokenwise_surrogate = (
        Fraction(6, 5) + Fraction(3, 4) - Fraction(4, 5) - Fraction(1)
    ) / 4
    surrogate_gradient_at_old = Fraction(1)
    learning_rate = Fraction(1, 10)
    optimizer_new_theta = old_theta + learning_rate * surrogate_gradient_at_old
    # New continuation applied to an old-writer mixture is .5*.6+.5*.2=.4.
    y = {"OO": old_return, "NO": Fraction(14, 25), "ON": Fraction(2, 5), "NN": new_return}
    writer = y["NO"] - y["OO"]
    continuation = y["ON"] - y["OO"]
    interaction = y["NN"] - y["NO"] - y["ON"] + y["OO"]
    closed = y["NN"] - y["OO"]
    # KL(old||new): writer plus old-writer-weighted continuation KL. The A
    # continuation is unchanged; B changes Bernoulli(.8) to Bernoulli(.6).
    writer_kl = .5 * math.log(.5/.6) + .5 * math.log(.5/.4)
    b_cont_kl = .8 * math.log(.8/.6) + .2 * math.log(.2/.4)
    trajectory_kl = writer_kl + .5 * b_cont_kl
    kl_aperture = .1
    ratios = [Fraction(6,5), Fraction(3,4), Fraction(4,5), Fraction(1)]
    clip_terms = [Fraction(6,5), Fraction(3,4), Fraction(-4,5), Fraction(-1)]
    passed = (optimizer_new_theta == new_theta and sampled_tokenwise_surrogate > 0
              and writer > 0 and closed < 0
              and closed == writer + continuation + interaction
              and trajectory_kl < kl_aperture)
    unsigned = {
        "schema": "memagent.cosi.e0.v1",
        "construction": "shared_theta_positive_sampled_clipped_ppo_surrogate_reversal",
        "old_theta": str(old_theta), "new_theta": str(new_theta),
        "grpo_group_rewards": [1, 0], "grpo_group_advantages": [1, -1],
        "surrogate_gradient_at_old": str(surrogate_gradient_at_old),
        "optimizer": "one_step_sgd_ascent", "learning_rate": str(learning_rate),
        "optimizer_new_theta": str(optimizer_new_theta),
        "clip_epsilon": str(clip_epsilon),
        "unclipped_ratios_at_new": [str(value) for value in ratios],
        "signed_clipped_terms_at_new": [str(value) for value in clip_terms],
        "trajectory_kl_old_to_new": trajectory_kl, "kl_aperture": kl_aperture,
        "kl_aperture_pass": trajectory_kl < kl_aperture,
        "sampled_tokenwise_surrogate": str(sampled_tokenwise_surrogate),
        "old_full_return": str(old_return), "new_full_return": str(new_return),
        "cells": {key: str(value) for key, value in y.items()},
        "writer_old": str(writer), "continuation_old": str(continuation),
        "interaction": str(interaction), "closed": str(closed),
        "exact_closure": closed == writer + continuation + interaction,
        "positive_sampled_surrogate": sampled_tokenwise_surrogate > 0,
        "positive_local_negative_closed": writer > 0 and closed < 0,
        "status": "PASS" if passed else "FAIL",
        "decision": "COSI_E0_PASS" if passed else "COSI_E0_NO_GO",
    }
    report = {**unsigned, "report_sha256": canonical_sha256(unsigned)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
