"""Build the versioned refractory/ceramic proxy dataset for Task 6B.

The rule is a generator-domain filter, not a label claiming high-temperature
performance.  Source data remains immutable; all outputs are derived files.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from pymatgen.core import Composition


ROOT = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = ROOT / "research" / "phase_2" / "data"
PROCESSED = DATA_DIR / "processed"
SOURCE_CSV = PROCESSED / "materials.csv"
SOURCE_GRAPHS = PROCESSED / "materials_graphs.pt"
VERSION = "high_temperature_proxy_v1"
OUT_CSV = PROCESSED / f"{VERSION}.csv"
OUT_GRAPHS = PROCESSED / f"{VERSION}_graphs.pt"
OUT_STATS = PROCESSED / f"{VERSION}_stats.json"
OUT_SPLIT = DATA_DIR / "splits" / f"{VERSION}_seed42.json"
SEED = 42

SYMBOLS = {
    5: "B", 6: "C", 7: "N", 8: "O", 13: "Al", 14: "Si",
    22: "Ti", 40: "Zr", 41: "Nb", 42: "Mo", 72: "Hf", 73: "Ta", 74: "W",
    1: "H", 9: "F", 17: "Cl", 35: "Br", 53: "I",
    3: "Li", 11: "Na", 19: "K", 37: "Rb", 55: "Cs",
}
REFRACTORY = {22, 40, 41, 42, 72, 73, 74}
CERAMIC = {5, 6, 7, 8, 13, 14}
FORBIDDEN = {1, 3, 9, 11, 17, 19, 35, 37, 53, 55}
MAX_NODES, MAX_EDGES, ENERGY_MAX = 64, 2048, -1.0


def reduced_group_key(formula: str, uid: str) -> str:
    try:
        return Composition(formula).reduced_formula
    except (ValueError, TypeError):
        return f"uid:{uid}"


def grouped_split(records: list[dict]) -> dict[str, list[str]]:
    """Deterministically keep every reduced-composition group in one split."""
    groups: dict[str, list[str]] = defaultdict(list)
    for row in records:
        groups[row["composition_group"]].append(row["material_uid"])
    targets = {"train": 0.80 * len(records), "val": 0.10 * len(records), "test": 0.10 * len(records)}
    assigned = {name: [] for name in targets}
    counts = {name: 0 for name in targets}
    ordered = sorted(
        groups.items(),
        key=lambda item: (-len(item[1]), hashlib.sha256(f"{SEED}:{item[0]}".encode()).hexdigest()),
    )
    for _, uids in ordered:
        split = max(targets, key=lambda name: targets[name] - counts[name])
        assigned[split].extend(sorted(uids))
        counts[split] += len(uids)
    return assigned


def main() -> None:
    graphs = torch.load(SOURCE_GRAPHS, map_location="cpu", weights_only=False)
    with SOURCE_CSV.open(newline="", encoding="utf-8") as handle:
        rows_by_index = {int(row["graph_index"]): row for row in csv.DictReader(handle)}

    selected_graphs, records = [], []
    rejected = Counter()
    for source_index, graph in enumerate(graphs):
        row = rows_by_index.get(source_index)
        if row is None:
            rejected["missing_csv_metadata"] += 1
            continue
        atomic_numbers = {int(number) for number in graph.x.view(-1).tolist()}
        energy = float(graph.y.view(-1)[0])
        reasons = []
        if graph.x.size(0) > MAX_NODES:
            reasons.append("max_nodes")
        if graph.edge_index.size(1) > MAX_EDGES:
            reasons.append("max_edges")
        if energy > ENERGY_MAX:
            reasons.append("formation_energy")
        if not atomic_numbers.intersection(REFRACTORY):
            reasons.append("missing_refractory")
        if not atomic_numbers.intersection(CERAMIC):
            reasons.append("missing_ceramic_former")
        if atomic_numbers.intersection(FORBIDDEN):
            reasons.append("forbidden_element")
        if reasons:
            rejected[";".join(reasons)] += 1
            continue

        refractory = sorted(SYMBOLS[number] for number in atomic_numbers.intersection(REFRACTORY))
        ceramic = sorted(SYMBOLS[number] for number in atomic_numbers.intersection(CERAMIC))
        uid = row["uid"]
        record = {
            "material_uid": uid,
            "source_graph_index": source_index,
            "domain_rule_version": VERSION,
            "refractory_elements": ";".join(refractory),
            "ceramic_forming_elements": ";".join(ceramic),
            "formation_energy_per_atom": energy,
            "domain_eligible": True,
            "rejection_reason": "",
            "composition_group": reduced_group_key(row["formula"], uid),
        }
        records.append(record)
        selected_graphs.append(graph)

    if not records:
        raise RuntimeError("Proxy rule selected zero graphs; check the source dataset.")
    split = grouped_split(records)
    split_by_uid = {uid: name for name, uids in split.items() for uid in uids}
    for record in records:
        record["split"] = split_by_uid[record["material_uid"]]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    OUT_SPLIT.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    torch.save(selected_graphs, OUT_GRAPHS)
    with OUT_SPLIT.open("w", encoding="utf-8") as handle:
        json.dump({"version": VERSION, "seed": SEED, "group_key": "reduced_formula", **split}, handle, indent=2)

    energies = [row["formation_energy_per_atom"] for row in records]
    stats = {
        "version": VERSION,
        "source_graphs": len(graphs),
        "eligible_graphs": len(records),
        "rule": {
            "refractory": [SYMBOLS[number] for number in sorted(REFRACTORY)],
            "ceramic_forming": [SYMBOLS[number] for number in sorted(CERAMIC)],
            "forbidden": [SYMBOLS[number] for number in sorted(FORBIDDEN)],
            "formation_energy_max_ev_per_atom": ENERGY_MAX,
            "max_nodes": MAX_NODES,
            "max_edges_directed": MAX_EDGES,
        },
        "split_counts": {name: len(uids) for name, uids in split.items()},
        "composition_groups": len({row["composition_group"] for row in records}),
        "energy_min": min(energies), "energy_max": max(energies),
        "rejection_counts": dict(rejected),
    }
    with OUT_STATS.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)
    print(f"[{VERSION}] selected {len(records)}/{len(graphs)} graphs")
    print(f"  split={stats['split_counts']}, groups={stats['composition_groups']}")
    print(f"  csv={OUT_CSV}\n  graphs={OUT_GRAPHS}\n  stats={OUT_STATS}\n  split={OUT_SPLIT}")


if __name__ == "__main__":
    main()
