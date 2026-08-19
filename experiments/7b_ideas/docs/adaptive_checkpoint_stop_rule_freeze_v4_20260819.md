# Adaptive checkpoint stop-rule freeze v4

The anchor schedule is frozen before any anchor value is unblinded: 2, 25, 50, 100, and terminal 200. T100 and every intermediate anchor are post-freeze screening only; `controls_continuation=false`. No accuracy, reward, loss, gradient, Shape A, or other scientific metric may stop or continue the run.

Early termination is permitted only for NaN, unrecoverable OOM, identity/data corruption, or resources outside existing authorization. Step 200 is terminal and is followed by stop. Step 400 requires a separate rule frozen before its own evidence is viewed and is never automatic here.

Confirmatory reporting cannot choose the best checkpoint among 25/50/100/200. Intermediate checkpoints remain labeled screening snapshots under the frozen terminal rule. This contract changes only launcher/reporting guards and authorizes no new training.
