"""
Diagnostic: inspect materials_graphs.pt for NaN/Inf values and extreme statistics.
Helps identify which graphs are causing NaN during VAE training.
"""

import torch
from pathlib import Path

# __file__ = /.../research/phase_2/generation/debug_data.py
#   .parent  = generation/
#   .parent.parent = phase_2/
#   .parent.parent.parent = research/
#   .parent.parent.parent.parent = ai-meterial/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GRAPH_PATH = PROJECT_ROOT / "phase_2" / "data" / "processed" / "materials_graphs.pt"

# Resolve symlinks on Mac (Downloads is often a symlink)
GRAPH_PATH = GRAPH_PATH.resolve()
PROJECT_ROOT = PROJECT_ROOT.resolve()
print(f"[DEBUG] PROJECT_ROOT = {PROJECT_ROOT}")
print(f"[DEBUG] GRAPH_PATH   = {GRAPH_PATH}")

if not GRAPH_PATH.exists():
    raise FileNotFoundError(f"Graph file not found at {GRAPH_PATH}")

print(f"Loading {GRAPH_PATH} ...")
graphs = torch.load(GRAPH_PATH, weights_only=False)
print(f"Total graphs: {len(graphs)}")

issues = []

for idx, g in enumerate(graphs):
    problems = []

    # Check x (atomic numbers)
    x = g.x
    if not torch.isfinite(x).all():
        problems.append(f"x NaN/Inf (min={x.min():.2f}, max={x.max():.2f})")
    if x.max() > 94:
        problems.append(f"x out of range (max={x.max():.0f}, >94)")
    if x.min() < 1:
        problems.append(f"x out of range (min={x.min():.0f}, <1)")

    # Check edge_attr (bond lengths)
    ea = g.edge_attr
    if not torch.isfinite(ea).all():
        problems.append(f"edge_attr NaN/Inf")
    if ea.max() > 10:
        problems.append(f"edge_attr extreme (max={ea.max():.2f} > 10 Å)")
    if ea.min() < 0:
        problems.append(f"edge_attr negative (min={ea.min():.2f})")
    if ea.mean() > 5:
        problems.append(f"edge_attr very large mean={ea.mean():.2f}")

    # Check y (formation energy)
    if hasattr(g, "y") and g.y is not None:
        y = g.y
        if not torch.isfinite(y).all():
            problems.append(f"y NaN/Inf")
        if torch.abs(y).max() > 10:
            problems.append(f"y extreme (max={y.max():.2f})")

    # Check graph size
    n_nodes = g.x.size(0)
    n_edges = g.edge_index.size(1)
    if n_nodes > 200:
        problems.append(f"very large graph: {n_nodes} nodes")

    if problems:
        uid = getattr(g, "material_uid", f"idx_{idx}")
        issues.append({
            "idx": idx,
            "uid": uid,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "x_min": float(x.min()),
            "x_max": float(x.max()),
            "ea_min": float(ea.min()),
            "ea_max": float(ea.max()),
            "ea_mean": float(ea.mean()),
            "problems": "; ".join(problems),
        })

print(f"\n{'='*80}")
print(f"Graphs with issues: {len(issues)}/{len(graphs)}")
print(f"{'='*80}")

if issues:
    print(f"\n{'idx':>6} | {'uid':<30} | {'nodes':>6} | {'edges':>6} | {'x_range':>12} | {'ea_range':>16} | Problems")
    print("-"*120)
    for iss in issues[:50]:
        print(
            f"{iss['idx']:>6} | {iss['uid']:<30} | "
            f"{iss['n_nodes']:>6} | {iss['n_edges']:>6} | "
            f"[{iss['x_min']:>5.0f},{iss['x_max']:>5.0f}] | "
            f"[{iss['ea_min']:>6.2f},{iss['ea_max']:>6.2f}] | "
            f"{iss['problems']}"
        )
    if len(issues) > 50:
        print(f"  ... and {len(issues)-50} more")

    # Summary of problem types
    print(f"\nProblem type summary:")
    from collections import Counter
    types = Counter()
    for iss in issues:
        for p in iss["problems"].split("; "):
            types[p.split("(")[0].strip()] += 1
    for t, c in types.most_common():
        print(f"  {t}: {c}")

    # Stats on clean graphs
    clean = [g for g in graphs if g.x.max() <= 94 and g.x.min() >= 1
             and torch.isfinite(g.x).all() and torch.isfinite(g.edge_attr).all()
             and g.edge_attr.max() <= 10 and g.edge_attr.min() >= 0]
    print(f"\nClean graphs: {len(clean)}/{len(graphs)}")
    if clean:
        ea_vals = torch.cat([g.edge_attr for g in clean])
        x_vals = torch.cat([g.x for g in clean])
        print(f"  x range:  [{x_vals.min():.0f}, {x_vals.max():.0f}]")
        print(f"  edge_attr range:  [{ea_vals.min():.4f}, {ea_vals.max():.4f}]")
        print(f"  edge_attr mean:   {ea_vals.mean():.4f}")
        print(f"  edge_attr std:    {ea_vals.std():.4f}")
        print(f"  edge_attr median: {ea_vals.median():.4f}")
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            val = torch.quantile(ea_vals, p / 100).item()
            print(f"  edge_attr p{p}: {val:.4f}")
else:
    print("No issues found — NaN must come from model initialization or training dynamics.")
    # Still show stats
    ea_vals = torch.cat([g.edge_attr for g in graphs])
    x_vals = torch.cat([g.x for g in graphs])
    print(f"\nx range:  [{x_vals.min():.0f}, {x_vals.max():.0f}]")
    print(f"edge_attr range:  [{ea_vals.min():.4f}, {ea_vals.max():.4f}]")
    print(f"edge_attr mean:   {ea_vals.mean():.4f}")
    print(f"edge_attr std:    {ea_vals.std():.4f}")
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    for p in percentiles:
        val = torch.quantile(ea_vals, p / 100).item()
        print(f"  edge_attr p{p}: {val:.4f}")
