#!/usr/bin/env python3
"""Runtime oracle for CORAL's zero-leaf DataProto capture path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path

import numpy as np
import tensordict
import torch
from tensordict import TensorDict

from verl import DataProto


def canonical_bytes(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise SystemExit("CORAL_E1_NO_GO:dataproto_clone_oracle_exists")

    source = DataProto(
        batch=TensorDict({}, batch_size=(8,)),
        non_tensor_batch={
            "uid": np.asarray([f"root-{index}" for index in range(8)], dtype=object),
            "index": np.arange(8, dtype=np.int64),
        },
        meta_info={"nested": {"phase": "source"}},
    )
    cloned = source.clone()
    if len(cloned) != 8 or list(cloned.batch.keys()):
        raise SystemExit("CORAL_E1_NO_GO:zero_leaf_batch_shape_drift")
    cloned.non_tensor_batch["uid"][0] = "mutated"
    cloned.meta_info["nested"]["phase"] = "mutated"
    if source.non_tensor_batch["uid"][0] != "root-0":
        raise SystemExit("CORAL_E1_NO_GO:non_tensor_clone_alias")
    if source.meta_info["nested"]["phase"] != "source":
        raise SystemExit("CORAL_E1_NO_GO:meta_clone_alias")

    tensor_source = DataProto(
        batch=TensorDict({"value": torch.arange(8)}, batch_size=(8,)),
        non_tensor_batch={},
        meta_info={},
    )
    tensor_clone = tensor_source.clone()
    tensor_clone.batch["value"][0] = -1
    if int(tensor_source.batch["value"][0]) != 0:
        raise SystemExit("CORAL_E1_NO_GO:tensor_clone_alias")

    report = {
        "schema": "memagent.coral.dataproto-clone-oracle.v1",
        "status": "PASS",
        "decision": "CORAL_DATAPROTO_CLONE_ORACLE_PASS",
        "batch_size": 8,
        "zero_leaf_keys": 0,
        "tensor_clone_independent": True,
        "non_tensor_clone_independent": True,
        "meta_clone_independent": True,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "tensordict_version": tensordict.__version__,
    }
    report["report_sha256"] = hashlib.sha256(canonical_bytes(report)).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    temporary.write_bytes(canonical_bytes(report) + b"\n")
    os.replace(temporary, output)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
