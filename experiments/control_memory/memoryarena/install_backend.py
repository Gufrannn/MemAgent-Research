#!/usr/bin/env python3
"""Install the two paired backends into an official MemoryArena checkout."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def replace_once(text: str, needle: str, replacement: str) -> str:
    if replacement in text:
        return text
    if needle not in text:
        raise RuntimeError(f"official source layout changed; missing: {needle!r}")
    return text.replace(needle, replacement, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("official_checkout", type=Path)
    args = parser.parse_args()
    root = args.official_checkout.resolve()
    target_dir = root / "memory" / "memory_systems"
    if not (root / "memory" / "server.py").exists():
        raise SystemExit(f"not an official MemoryArena checkout: {root}")

    source = Path(__file__).with_name("control_boundary.py")
    shutil.copy2(source, target_dir / source.name)
    config_dir = root / "configs" / "web_shopping_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    for config_name in ("control_raw-qwen25-7b.json", "control_boundary-qwen25-7b.json"):
        shutil.copy2(Path(__file__).with_name(config_name), config_dir / config_name)

    init_path = target_dir / "__init__.py"
    init_text = init_path.read_text()
    init_text = replace_once(
        init_text,
        "from .zep import ZepMemorySystem",
        "from .zep import ZepMemorySystem\nfrom .control_boundary import ControlBoundaryMemorySystem",
    )
    init_path.write_text(init_text)

    server_path = root / "memory" / "server.py"
    server_text = server_path.read_text()
    server_text = replace_once(
        server_text,
        "    ZepMemorySystem,\n)",
        "    ZepMemorySystem,\n    ControlBoundaryMemorySystem,\n)",
    )
    server_text = replace_once(
        server_text,
        '    "zep": ZepMemorySystem,',
        '    "zep": ZepMemorySystem,\n'
        '    "control_raw": lambda: ControlBoundaryMemorySystem(mode="raw"),\n'
        '    "control_boundary": lambda: ControlBoundaryMemorySystem(mode="control"),',
    )
    server_path.write_text(server_text)
    print(f"installed control_raw, control_boundary, and paired WebShop configs into {root}")


if __name__ == "__main__":
    main()
