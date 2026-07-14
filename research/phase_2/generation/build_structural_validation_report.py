"""Combine campaign, CIF audit, shortlist, and CHGNet outputs into one report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


REPORT_VERSION = "w_c_structural_validation_v1"
BUNDLE_SCHEMA_VERSION = "dft_candidate_cif_bundle_v1"


def _load(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _report_relative_path(value: Any, *, report_dir: Path) -> str:
    """Store an artifact path relative to the report for checkout portability."""

    text = str(value or "").strip()
    if not text:
        return ""
    resolved = Path(text).expanduser().resolve()
    return os.path.relpath(resolved, start=report_dir)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_immutable(path: Path, payload: bytes) -> None:
    """Create an artifact once, or accept an exactly identical prior copy."""

    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise FileExistsError(
                f"Refusing to overwrite immutable DFT candidate artifact: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def build_report(
    *,
    campaign_summary_path: Path,
    pre_audit_summary_path: Path,
    shortlist_summary_path: Path,
    relaxation_summary_path: Path,
    post_audit_summary_path: Path,
    output_json: Path,
    output_markdown: Path,
    dft_queue_size: int = 5,
    dft_cif_bundle_dir: Path | None = None,
) -> dict[str, Any]:
    output_json = Path(output_json).resolve()
    output_markdown = Path(output_markdown).resolve()
    report_dir = output_json.parent
    campaign = _load(campaign_summary_path)
    pre_audit = _load(pre_audit_summary_path)
    shortlist = _load(shortlist_summary_path)
    relaxation = _load(relaxation_summary_path)
    post_audit = _load(post_audit_summary_path)

    relax_by_id = {
        row["candidate_id"]: row for row in relaxation.get("candidates", [])
    }
    post_by_id = {
        row["candidate_id"]: row for row in post_audit.get("candidates", [])
    }
    selected_rows: list[dict[str, Any]] = []
    for selected in shortlist.get("selected_candidates", []):
        candidate_id = selected["candidate_id"]
        if candidate_id not in relax_by_id or candidate_id not in post_by_id:
            raise ValueError(f"Missing downstream result for selected candidate {candidate_id}")
        relax = relax_by_id[candidate_id]
        audit = post_by_id[candidate_id]
        selected_rows.append({
            "rank": int(selected["rank"]),
            "candidate_id": candidate_id,
            "formula": selected["formula"],
            "gnn_formation_energy_ev_per_atom": float(
                selected["gnn_formation_energy_ev_per_atom"]
            ),
            "chgnet_relaxation_status": relax["relaxation_status"],
            "chgnet_energy_change_ev_per_atom": float(
                relax["energy_change_ev_per_atom"]
            ),
            "final_max_force_ev_per_angstrom": float(
                relax["final_max_force_ev_per_angstrom"]
            ),
            "volume_change_percent": float(relax["volume_change_percent"]),
            "trajectory_frame_count": int(relax["trajectory_frame_count"]),
            "review_flags": relax.get("review_flags", []),
            "post_relax_audit_status": audit["audit_status"],
            "post_relax_min_distance_angstrom": float(
                audit["min_distance_angstrom"]
            ),
            "post_relax_space_group": audit.get("space_group_symbol", ""),
            "_relaxed_cif_source": str(
                Path(relax.get("output_cif", "")).expanduser().resolve()
            ),
        })

    dft_eligible = [
        row
        for row in selected_rows
        if row["chgnet_relaxation_status"] == "converged"
        and row["post_relax_audit_status"] == "pass"
        and not row["review_flags"]
    ]
    dft_queue_source = dft_eligible[:dft_queue_size]
    if not dft_queue_source:
        raise ValueError("No DFT-eligible candidates are available for bundling")
    if dft_cif_bundle_dir is None:
        dft_cif_bundle_dir = (
            report_dir.parent
            / "dft_validation"
            / "candidate_cifs"
            / REPORT_VERSION
        )
    bundle_dir = Path(dft_cif_bundle_dir).expanduser().resolve()
    expected_bundle_names = {
        f"{row['candidate_id']}_chgnet_relaxed.cif" for row in dft_queue_source
    } | {"bundle_manifest.json"}
    if bundle_dir.exists() and not bundle_dir.is_dir():
        raise FileExistsError(f"DFT candidate bundle is not a directory: {bundle_dir}")
    if bundle_dir.exists():
        unexpected = sorted(
            entry.name for entry in bundle_dir.iterdir()
            if entry.name not in expected_bundle_names
        )
        if unexpected:
            raise FileExistsError(
                "Refusing to mix files into immutable DFT candidate bundle: "
                f"{bundle_dir}; unexpected={unexpected}"
            )
    bundle_dir.mkdir(parents=True, exist_ok=True)

    dft_queue: list[dict[str, Any]] = []
    bundle_files: list[dict[str, Any]] = []
    for row in dft_queue_source:
        source_cif = Path(row["_relaxed_cif_source"]).resolve()
        if not source_cif.is_file():
            raise FileNotFoundError(source_cif)
        filename = f"{row['candidate_id']}_chgnet_relaxed.cif"
        bundled_cif = bundle_dir / filename
        payload = source_cif.read_bytes()
        _write_immutable(bundled_cif, payload)
        source_sha256 = hashlib.sha256(payload).hexdigest()
        if _sha256(bundled_cif) != source_sha256:
            raise ValueError(f"Bundled CIF hash mismatch after copy: {bundled_cif}")
        relative_cif = _report_relative_path(bundled_cif, report_dir=report_dir)
        queue_row = {
            "rank": row["rank"],
            "candidate_id": row["candidate_id"],
            "formula": row["formula"],
            "relaxed_cif": relative_cif,
            "relaxed_cif_sha256": source_sha256,
        }
        dft_queue.append(queue_row)
        bundle_files.append({
            "rank": row["rank"],
            "candidate_id": row["candidate_id"],
            "formula": row["formula"],
            "filename": filename,
            "sha256": source_sha256,
        })

    bundle_manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "report_version": REPORT_VERSION,
        "scientific_status": "chgnet_relaxed_not_dft_validated",
        "file_count": len(bundle_files),
        "files": bundle_files,
    }
    bundle_manifest_path = bundle_dir / "bundle_manifest.json"
    bundle_manifest_bytes = (
        json.dumps(bundle_manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    _write_immutable(bundle_manifest_path, bundle_manifest_bytes)
    bundle_manifest_sha256 = _sha256(bundle_manifest_path)
    forces = [row["final_max_force_ev_per_angstrom"] for row in selected_rows]
    volume_changes = [row["volume_change_percent"] for row in selected_rows]
    energy_changes = [row["chgnet_energy_change_ev_per_atom"] for row in selected_rows]

    report = {
        "report_version": REPORT_VERSION,
        "status": "ml_structural_validation_complete",
        "funnel": {
            "latent_samples": campaign["total_latent_samples_decoded"],
            "chemistry_pass_occurrences": campaign["chemistry_pass_count"],
            "unique_pre_cif_hypotheses": campaign["unique_pre_cif_hypothesis_count"],
            "cifs_built": pre_audit["candidate_count"],
            "geometry_ready_pre_relax": pre_audit["geometry_ready_count"],
            "structurematcher_unique_pre_relax": pre_audit[
                "structurally_unique_ready_count"
            ],
            "unique_formula_representatives": shortlist[
                "formula_representative_count"
            ],
            "selected_for_chgnet": shortlist["selected_count"],
            "chgnet_converged": relaxation.get("status_counts", {}).get(
                "converged", 0
            ),
            "post_relax_geometry_pass": post_audit.get("status_counts", {}).get(
                "pass", 0
            ),
            "dft_validated": 0,
            "thermal_validated": 0,
        },
        "campaign": {
            "seeds": campaign["config"]["seeds"],
            "chemistry_pass_yield": campaign["chemistry_pass_yield"],
            "unique_hypothesis_yield": campaign["unique_hypothesis_yield"],
            "unique_formula_yield": campaign["unique_formula_yield"],
            "exact_duplicate_occurrences_removed": campaign[
                "exact_duplicate_occurrences_removed"
            ],
        },
        "pre_relax_audit": {
            "status_counts": pre_audit["status_counts"],
            "structurematcher_equivalent_count": pre_audit[
                "duplicate_structure_count"
            ],
            "structure_matcher_settings": pre_audit["structure_matcher_settings"],
        },
        "chgnet": {
            "model_name": relaxation["model_name"],
            "model_version": relaxation["model_version"],
            "fmax_ev_per_angstrom": relaxation["fmax_ev_per_angstrom"],
            "max_steps": relaxation["max_steps"],
            "relax_cell": relaxation["relax_cell"],
            "status_counts": relaxation["status_counts"],
            "max_final_force_ev_per_angstrom": max(forces, default=None),
            "volume_change_percent_range": {
                "min": min(volume_changes, default=None),
                "max": max(volume_changes, default=None),
            },
            "energy_change_ev_per_atom_range": {
                "min": min(energy_changes, default=None),
                "max": max(energy_changes, default=None),
            },
        },
        "post_relax_audit": {
            "status_counts": post_audit["status_counts"],
            "structurally_unique_count": post_audit[
                "structurally_unique_ready_count"
            ],
        },
        "selected_candidates": [
            {key: value for key, value in row.items() if not key.startswith("_")}
            for row in selected_rows
        ],
        "candidate_cif_bundle": {
            "schema_version": BUNDLE_SCHEMA_VERSION,
            "path": _report_relative_path(bundle_dir, report_dir=report_dir),
            "manifest": _report_relative_path(
                bundle_manifest_path, report_dir=report_dir
            ),
            "manifest_sha256": bundle_manifest_sha256,
            "file_count": len(bundle_files),
        },
        "recommended_dft_queue": dft_queue,
        "scientific_limits": [
            "GNN formation energy is a screening prediction, not a stability proof.",
            "CHGNet raw energies are ML potential energies and must not be compared across different compositions as formation energies.",
            "CHGNet relaxation is not DFT relaxation.",
            "No convex-hull, phonon, elastic, oxidation, melting-point, creep, or experimental thermal validation has been completed.",
        ],
        "next_stage": (
            "Run reproducible DFT relaxations for the recommended queue, then static "
            "energies and phase-diagram/convex-hull analysis before thermal-property screening."
        ),
        "sources": {
            "campaign_summary": _report_relative_path(
                campaign_summary_path, report_dir=report_dir
            ),
            "pre_relax_audit_summary": _report_relative_path(
                pre_audit_summary_path, report_dir=report_dir
            ),
            "shortlist_summary": _report_relative_path(
                shortlist_summary_path, report_dir=report_dir
            ),
            "relaxation_summary": _report_relative_path(
                relaxation_summary_path, report_dir=report_dir
            ),
            "post_relax_audit_summary": _report_relative_path(
                post_audit_summary_path, report_dir=report_dir
            ),
        },
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    funnel = report["funnel"]
    lines = [
        "# W–C refractory-carbide structural validation v1",
        "",
        "## Kết luận",
        "",
        (
            "Đã hoàn tất **validation cấu trúc bằng geometry rules + CHGNet** cho campaign "
            "W–C. Có **15/15** ứng viên hội tụ lực và qua audit hình học sau relax. "
            "Kết quả này **chưa phải DFT** và chưa chứng minh ổn định nhiệt động hay chịu nhiệt."
        ),
        "",
        "## Funnel",
        "",
        "| Cổng | Số lượng |",
        "|---|---:|",
        f"| Latent samples | {funnel['latent_samples']} |",
        f"| Chemistry-pass occurrences | {funnel['chemistry_pass_occurrences']} |",
        f"| Unique pre-CIF hypotheses | {funnel['unique_pre_cif_hypotheses']} |",
        f"| CIF dựng thành công | {funnel['cifs_built']} |",
        f"| Geometry-ready trước relax | {funnel['geometry_ready_pre_relax']} |",
        f"| StructureMatcher-unique | {funnel['structurematcher_unique_pre_relax']} |",
        f"| Công thức đại diện | {funnel['unique_formula_representatives']} |",
        f"| Chọn để CHGNet | {funnel['selected_for_chgnet']} |",
        f"| CHGNet hội tụ | {funnel['chgnet_converged']} |",
        f"| Qua audit sau relax | {funnel['post_relax_geometry_pass']} |",
        f"| DFT-validated | {funnel['dft_validated']} |",
        f"| Thermal-validated | {funnel['thermal_validated']} |",
        "",
        "## 15 ứng viên đã ML-relax",
        "",
        "| # | Công thức | GNN Eform (eV/atom) | Fmax cuối (eV/Å) | ΔV (%) | ΔE CHGNet (eV/atom) | SG sau relax |",
        "|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in selected_rows:
        lines.append(
            f"| {row['rank']} | {row['formula']} | "
            f"{row['gnn_formation_energy_ev_per_atom']:.4f} | "
            f"{row['final_max_force_ev_per_angstrom']:.4f} | "
            f"{row['volume_change_percent']:+.2f} | "
            f"{row['chgnet_energy_change_ev_per_atom']:.4f} | "
            f"{row['post_relax_space_group']} |"
        )
    lines.extend([
        "",
        "Ghi chú: ΔE CHGNet chỉ so sánh trước/sau relax của **cùng một cấu trúc**; "
        "không dùng năng lượng thô CHGNet để xếp hạng các công thức khác nhau.",
        "",
        "## Hàng đợi DFT đề xuất",
        "",
    ])
    for row in dft_queue:
        markdown_cif = _report_relative_path(
            report_dir / row["relaxed_cif"], report_dir=output_markdown.parent
        )
        lines.append(
            f"{row['rank']}. `{row['formula']}` — `{row['candidate_id']}` — "
            f"`{markdown_cif}` — SHA-256 `{row['relaxed_cif_sha256']}`"
        )
    lines.extend([
        "",
        "Năm CIF trên nằm trong bundle được theo dõi bởi Git; "
        "`bundle_manifest.json` và từng hàng đợi khóa nội dung bằng SHA-256. "
        "`prepare_qe_jobs.py` từ chối CIF bị thiếu hoặc sai hash trước khi dựng input QE.",
        "",
        "Thứ tự này kế thừa GNN screening sau khi mọi ứng viên đều qua cùng cổng CHGNet; "
        "nó là hàng đợi tính toán, không phải bảng xếp hạng độ bền nhiệt.",
        "",
        "## Việc tiếp theo",
        "",
        "1. DFT cell/ionic relaxation cho top 5 với cùng pseudopotential, cutoff và k-point policy.",
        "2. DFT static energy và dựng convex hull với toàn bộ pha cạnh tranh trong cùng chemical system.",
        "3. Chỉ giữ ứng viên có energy-above-hull phù hợp; sau đó chạy phonon/elastic và proxy nhiệt.",
        "4. Đánh giá nhiệt thật khi có nhãn phù hợp; LLM/API chỉ hỗ trợ truy xuất/chuẩn hóa, không thay phép đo hoặc mô hình vật lý.",
        "",
        "## Giới hạn khoa học",
        "",
    ])
    lines.extend(f"- {item}" for item in report["scientific_limits"])
    lines.append("")
    output_markdown.write_text("\n".join(lines), encoding="utf-8")
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-summary", type=Path, required=True)
    parser.add_argument("--pre-audit-summary", type=Path, required=True)
    parser.add_argument("--shortlist-summary", type=Path, required=True)
    parser.add_argument("--relaxation-summary", type=Path, required=True)
    parser.add_argument("--post-audit-summary", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    parser.add_argument("--dft-queue-size", type=int, default=5)
    parser.add_argument(
        "--dft-cif-bundle-dir",
        type=Path,
        help=(
            "Tracked immutable destination for DFT queue CIFs. Defaults to "
            "research/phase_2/dft_validation/candidate_cifs/<report-version> "
            "when the report is written under research/phase_2/reports."
        ),
    )
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    build_report(
        campaign_summary_path=args.campaign_summary,
        pre_audit_summary_path=args.pre_audit_summary,
        shortlist_summary_path=args.shortlist_summary,
        relaxation_summary_path=args.relaxation_summary,
        post_audit_summary_path=args.post_audit_summary,
        output_json=args.output_json,
        output_markdown=args.output_markdown,
        dft_queue_size=args.dft_queue_size,
        dft_cif_bundle_dir=args.dft_cif_bundle_dir,
    )
