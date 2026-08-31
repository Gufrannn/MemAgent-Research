#!/usr/bin/env python3
"""Convert UMA response JSONL into official LongMemEval hypothesis JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def raw_qid(qid: str) -> str:
    qid = str(qid)
    return qid[len("longmemeval_") :] if qid.startswith("longmemeval_") else qid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.responses.open(encoding="utf-8") as input_file, args.output.open("w", encoding="utf-8") as output_file:
        for line in input_file:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = row.get("qid") or row.get("question_id")
            output_file.write(
                json.dumps(
                    {
                        "question_id": raw_qid(str(qid)),
                        "hypothesis": str(row.get("response") or ""),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n += 1
    print(json.dumps({"responses": str(args.responses), "output": str(args.output), "n": n}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
