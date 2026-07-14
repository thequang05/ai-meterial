"""Create a reproducible, composition-grouped broad split for Phase 2 models."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from pymatgen.core import Composition


ROOT = Path(__file__).resolve().parent.parent.parent
CSV_PATH = ROOT / "research" / "phase_2" / "data" / "processed" / "materials.csv"
OUT_PATH = ROOT / "research" / "phase_2" / "data" / "splits" / "materials_broad_v1_seed42.json"
SEED = 42


def reduced_formula(formula: str, uid: str) -> str:
    try:
        return Composition(formula).reduced_formula
    except (ValueError, TypeError):
        return f"uid:{uid}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or len({row["uid"] for row in rows}) != len(rows):
        raise ValueError("materials.csv must contain one unique UID per graph.")

    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        groups[reduced_formula(row["formula"], row["uid"])].append(row["uid"])

    targets = {"train": 0.80 * len(rows), "val": 0.10 * len(rows), "test": 0.10 * len(rows)}
    splits = {name: [] for name in targets}
    counts = {name: 0 for name in targets}
    ordered_groups = sorted(
        groups.items(),
        key=lambda item: (-len(item[1]), hashlib.sha256(f"{SEED}:{item[0]}".encode()).hexdigest()),
    )
    for _, uids in ordered_groups:
        split = max(targets, key=lambda name: targets[name] - counts[name])
        splits[split].extend(sorted(uids))
        counts[split] += len(uids)

    all_uids = splits["train"] + splits["val"] + splits["test"]
    if len(all_uids) != len(rows) or len(set(all_uids)) != len(rows):
        raise RuntimeError("Split construction lost or duplicated UIDs.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "materials_broad_v1",
        "seed": SEED,
        "group_key": "reduced_formula",
        "source": {
            "csv": str(CSV_PATH.relative_to(ROOT)),
            "csv_sha256": sha256(CSV_PATH),
            "rows": len(rows),
            "composition_groups": len(groups),
        },
        **splits,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {OUT_PATH}")
    print(f"split counts={counts}; composition_groups={len(groups)}")


if __name__ == "__main__":
    main()
