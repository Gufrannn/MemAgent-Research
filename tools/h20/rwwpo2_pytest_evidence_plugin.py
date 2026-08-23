"""Pytest plugin that freezes exact collected node IDs and phase outcomes."""
from __future__ import annotations

import json
from pathlib import Path


_COLLECTED: list[str] = []
_REPORTS: dict[str, list[dict]] = {}


def pytest_addoption(parser) -> None:
    group = parser.getgroup("rwwpo2-evidence")
    group.addoption("--rwwpo2-evidence-json", required=True)
    group.addoption(
        "--rwwpo2-evidence-mode", choices=("collect", "execute"), required=True
    )


def pytest_collection_finish(session) -> None:
    global _COLLECTED
    _COLLECTED = [item.nodeid for item in session.items]


def pytest_runtest_logreport(report) -> None:
    _REPORTS.setdefault(report.nodeid, []).append({
        "when": report.when,
        "outcome": report.outcome,
        "wasxfail": bool(getattr(report, "wasxfail", False)),
    })


def pytest_sessionfinish(session, exitstatus: int) -> None:
    output = Path(session.config.getoption("--rwwpo2-evidence-json"))
    row = {
        "schema_version": "rwwpo2-pytest-node-evidence-v1",
        "mode": session.config.getoption("--rwwpo2-evidence-mode"),
        "pytest_exitstatus": int(exitstatus),
        "collected_node_ids": list(_COLLECTED),
        "phase_reports": {
            nodeid: list(_REPORTS.get(nodeid, [])) for nodeid in _COLLECTED
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, indent=2) + "\n")
