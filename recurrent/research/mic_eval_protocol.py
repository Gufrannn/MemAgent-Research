"""Checkout-neutral protocol binding for MIC fixed-S128 evaluation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any


PROJECTION_SCHEMA = "repository-relative-reward-code-sha256-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_neutral_generation_protocol(
    projection: Mapping[str, Any],
    *,
    repo_dir: str | Path,
    expected_reward_code_sha256: str,
) -> tuple[dict[str, Any], Path]:
    """Mirror the certified Original curve's repository-neutral projection.

    The Original producer removes only the checkout prefix from the reward
    module path, then binds both its repository-relative path and byte digest.
    Reward implementations outside the bound repository fail closed.
    """
    normalized = deepcopy(dict(projection))
    reward = dict(normalized["custom_reward_function"])
    reward_path = Path(str(reward.get("path", ""))).resolve()
    repo_root = Path(repo_dir).resolve()
    try:
        relative = reward_path.relative_to(repo_root)
    except ValueError as error:
        raise RuntimeError(
            f"MIC_NO_GO: Method reward path is outside the bound repository: "
            f"{reward_path}"
        ) from error
    if not reward_path.is_file():
        raise RuntimeError("MIC_NO_GO: Method evaluation reward code is absent")
    actual_reward_sha = _sha256_file(reward_path)
    if actual_reward_sha != expected_reward_code_sha256:
        raise RuntimeError("MIC_NO_GO: Method/Original S128 reward code differs")
    reward["path"] = relative.as_posix()
    reward["path_sha256"] = actual_reward_sha
    normalized["custom_reward_function"] = reward
    return normalized, reward_path
