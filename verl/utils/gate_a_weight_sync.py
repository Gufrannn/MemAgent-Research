"""Canonical sampled-tensor digests for actor-to-vLLM synchronization audits."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def evenly_spaced_indices(numel: int, sample_count: int) -> list[int]:
    if numel < 1 or sample_count < 1:
        raise ValueError(f"numel and sample_count must be positive: {numel=}, {sample_count=}")
    count = min(numel, sample_count)
    if count == 1:
        return [0]
    return [(index * (numel - 1)) // (count - 1) for index in range(count)]


def digest_sample_records(records: Iterable[tuple[str, Sequence[int], Sequence[int], bytes]]) -> str:
    digest = hashlib.sha256()
    for name, shape, indices, values in records:
        header = json.dumps(
            {"name": name, "shape": list(shape), "indices": list(indices), "dtype": "float32", "byte_order": "little"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(len(values).to_bytes(8, "little"))
        digest.update(values)
    return digest.hexdigest()


def _resolve_tensor(named_tensors: Mapping[str, Any], required_name: str) -> Any:
    if required_name in named_tensors:
        return named_tensors[required_name]
    matches = [(name, tensor) for name, tensor in named_tensors.items() if name.endswith(required_name)]
    if len(matches) != 1:
        raise KeyError(f"required sampled tensor is not uniquely resolvable: {required_name}; matches={[name for name, _ in matches]}")
    return matches[0][1]


def sampled_tensor_digest(named_tensors: Mapping[str, Any], parameter_names: Sequence[str], samples_per_tensor: int) -> str:
    import numpy as np
    import torch

    records = []
    for name in sorted(parameter_names):
        tensor = _resolve_tensor(named_tensors, name)
        if hasattr(tensor, "full_tensor"):
            tensor = tensor.full_tensor()
        tensor = tensor.detach().reshape(-1)
        indices = evenly_spaced_indices(tensor.numel(), samples_per_tensor)
        index_tensor = torch.tensor(indices, dtype=torch.long, device=tensor.device)
        values = tensor.index_select(0, index_tensor).to(dtype=torch.float32).cpu().contiguous().numpy()
        values = np.asarray(values, dtype="<f4").tobytes(order="C")
        records.append((name, tuple(_resolve_tensor(named_tensors, name).shape), indices, values))
    return digest_sample_records(records)
