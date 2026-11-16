"""Manually export human-reviewed candidates without modifying training data."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def export_reviewed(queue: Path, output: Path) -> int:
    records = [
        json.loads(line)
        for line in (queue / "candidates.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    selected = [
        item
        for item in records
        if item["review_status"] in {"reviewed", "accepted"} and item["human_verdict"]
    ]
    output.mkdir(parents=True, exist_ok=True)
    exported = []
    for item in selected:
        source = (
            queue / "images" / Path(item["artifact_reference"]).name
            if item["artifact_reference"]
            else None
        )
        if source and source.is_file():
            target = output / source.name
            shutil.copy2(source, target)
            item["exported_artifact"] = target.name
        exported.append(item)
    (output / "reviewed_candidates.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in exported), encoding="utf-8"
    )
    return len(exported)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    count = export_reviewed(args.queue, args.output)
    print(json.dumps({"exported": count, "training_manifest_modified": False}))


if __name__ == "__main__":
    main()
