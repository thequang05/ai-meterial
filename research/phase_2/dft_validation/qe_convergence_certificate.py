"""Create and verify fail-closed QE convergence integrity certificates.

The normal workflow creates this content-hash-bound receipt only after the
independent confirmation collector passes.  It binds the sweep, confirmation,
pseudopotentials, workflow configuration, protocol, QE version, and executable
against accidental or post-collection mutation.  It is deliberately *not* a
digital signature or an authentication mechanism: the workflow assumes a
trusted operator and trusted workspace.  Production relax/static preparation
imports this verifier so that a missing or modified artifact cannot silently
unlock calculations.
"""

from __future__ import annotations

import hashlib
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

from qe_execution_provenance import (
    require_same_execution_provenance,
    validate_execution_provenance,
)
from qe_convergence_evidence import (
    PRODUCTION_TRANSFER_SCOPE,
    derive_strictest_tested_settings,
    verify_provisional_sweep_evidence,
)

SCHEMA_VERSION = "qe_convergence_certificate_v1"
HASH_RE = re.compile(r"[0-9a-f]{64}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def issue_convergence_certificate(
    *, payload: dict[str, Any], output_path: Path,
) -> dict[str, Any]:
    """Write a canonical certificate around an already validated payload."""
    payload_sha = canonical_digest(payload)
    certificate = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed_tested_window",
        "certificate_eligible": True,
        "confirmation_required": False,
        "certificate_id": f"qeconv-{payload_sha[:20]}",
        "certificate_payload": payload,
        "certificate_payload_sha256": payload_sha,
        "scientific_limit": (
            "This certificate establishes numerical convergence only within the "
            "tested scalar-relativistic, non-spin, finite-smearing PBE window. "
            "It is not evidence of thermodynamic, dynamic, mechanical, oxidation, "
            "or high-temperature service stability."
        ),
    }
    output_path = Path(output_path).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite a convergence certificate: {output_path}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(certificate, indent=2), encoding="utf-8")
    return certificate


def _positive_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Certificate {label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Certificate {label} must be numeric") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"Certificate {label} must be finite and positive")
    return result


def _verify_locked_artifact(record: Any, label: str) -> dict[str, str]:
    if not isinstance(record, dict):
        raise ValueError(f"Certificate artifact record is missing: {label}")
    path_text = str(record.get("path") or "").strip()
    expected = str(record.get("sha256") or "").strip().lower()
    if not path_text or not HASH_RE.fullmatch(expected):
        raise ValueError(f"Certificate artifact record is invalid: {label}")
    path = Path(path_text).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Certificate artifact is missing: {label}:{path}")
    if sha256_file(path) != expected:
        raise ValueError(f"Certificate artifact hash mismatch: {label}")
    return {"path": str(path), "sha256": expected}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Certificate CSV artifact is empty: {path}")
    return rows


def _same_float(left: Any, right: Any, label: str, *, tolerance: float = 1e-10) -> None:
    first = _positive_float(left, label)
    second = _positive_float(right, label)
    if not math.isclose(first, second, rel_tol=1e-12, abs_tol=tolerance):
        raise ValueError(f"Certificate semantic lineage mismatch: {label}")


def _verify_semantic_lineage(
    *, payload: dict[str, Any], artifacts: dict[str, dict[str, str]],
) -> None:
    """Re-derive certificate eligibility from the locked JSON/CSV artifacts."""
    sweep_preflight = _load_json(Path(artifacts["sweep_preflight"]["path"]))
    sweep_summary = _load_json(Path(artifacts["sweep_summary"]["path"]))
    confirmation_preflight = _load_json(
        Path(artifacts["confirmation_preflight"]["path"])
    )
    confirmation_summary = _load_json(
        Path(artifacts["confirmation_summary"]["path"])
    )
    if (
        sweep_preflight.get("workflow_version") != "qe_pbe_convergence_sweep_v1"
        or sweep_preflight.get("stage") != "sweep"
        or sweep_preflight.get("calculation") != "scf"
        or sweep_preflight.get("coverage_complete") is not True
        or sweep_preflight.get("confirmation_eligible") is not True
        or sweep_preflight.get("certificate_eligible") is not False
    ):
        raise ValueError("Certificate sweep preflight is not confirmation-eligible")
    sweep_settings_payload = sweep_preflight.get("convergence_settings_payload")
    sweep_settings_hash = str(
        sweep_preflight.get("convergence_settings_hash") or ""
    ).lower()
    if (
        not isinstance(sweep_settings_payload, dict)
        or not HASH_RE.fullmatch(sweep_settings_hash)
        or canonical_digest(sweep_settings_payload) != sweep_settings_hash
        or sweep_settings_hash != payload.get("source_sweep_settings_hash")
    ):
        raise ValueError("Certificate sweep settings lineage is invalid")
    if (
        sweep_preflight.get("required_elements") != payload.get("required_elements")
        or sweep_preflight.get("covered_elements") != payload.get("covered_elements")
    ):
        raise ValueError("Certificate sweep element lineage mismatch")
    _same_float(
        sweep_preflight.get("base_ecutwfc_ry"), payload.get("base_ecutwfc_ry"),
        "base_ecutwfc_ry",
    )
    _same_float(
        sweep_preflight.get("base_ecutrho_ry"), payload.get("base_ecutrho_ry"),
        "base_ecutrho_ry", tolerance=1e-8,
    )
    for key, artifact_label, hash_key in (
        ("config", "config", "config_sha256"),
        ("pseudo_manifest", "pseudo_manifest", "pseudo_manifest_sha256"),
        ("protocol", "protocol", "protocol_sha256"),
    ):
        sweep_path = Path(str(sweep_preflight.get(key) or "")).resolve()
        expected = str(sweep_settings_payload.get(hash_key) or "").lower()
        if (
            not sweep_path.is_file()
            or not HASH_RE.fullmatch(expected)
            or sha256_file(sweep_path) != expected
            or expected != artifacts[artifact_label]["sha256"]
        ):
            raise ValueError(f"Certificate sweep {key} lineage mismatch")
    source_preflight_path = Path(
        str(sweep_preflight.get("source_preflight") or "")
    ).resolve()
    source_preflight_sha = str(
        sweep_preflight.get("source_preflight_sha256") or ""
    ).lower()
    if (
        not source_preflight_path.is_file()
        or not HASH_RE.fullmatch(source_preflight_sha)
        or sha256_file(source_preflight_path) != source_preflight_sha
    ):
        raise ValueError("Certificate convergence-source preflight hash mismatch")
    source_preflight = _load_json(source_preflight_path)
    if source_preflight.get("status") not in {
        "convergence_source_only_not_runnable",
        "blocked_missing_convergence_certificate",
    } or source_preflight.get("production_settings_certified") is not False:
        raise ValueError("Certificate convergence source was not a safe bootstrap queue")
    source_queue_path = Path(
        str(source_preflight.get("queue_manifest") or "")
    ).resolve()
    source_queue_sha = str(
        source_preflight.get("queue_manifest_sha256") or ""
    ).lower()
    if (
        not source_queue_path.is_file()
        or not HASH_RE.fullmatch(source_queue_sha)
        or sha256_file(source_queue_path) != source_queue_sha
        or source_queue_sha != sweep_preflight.get("source_queue_manifest_sha256")
    ):
        raise ValueError("Certificate convergence-source queue hash mismatch")

    required_sweep_state = {
        "collector_version": "qe_convergence_collector_v1",
        "status": "provisional_selection_ready",
        "confirmation_eligible": True,
        "certificate_eligible": False,
        "confirmation_required": True,
        "coverage_complete": True,
        "cutoff_window_converged": True,
        "kpoint_window_converged": True,
    }
    if any(sweep_summary.get(key) != value for key, value in required_sweep_state.items()):
        raise ValueError("Certificate provisional sweep summary is not eligible")
    if sweep_summary.get("global_failures") != []:
        raise ValueError("Certificate provisional sweep contains global failures")
    if (
        sweep_summary.get("convergence_settings_hash") != sweep_settings_hash
        or sweep_summary.get("convergence_preflight_sha256")
        != artifacts["sweep_preflight"]["sha256"]
        or sweep_summary.get("results_sha256")
        != artifacts["sweep_results"]["sha256"]
    ):
        raise ValueError("Certificate provisional sweep artifact lineage mismatch")
    selected = payload["selected_settings"]
    provisional = sweep_summary.get("provisional_global_selection")
    if not isinstance(provisional, dict):
        raise ValueError("Certificate provisional global selection is missing")
    for key in (
        "cutoff_pair_multiplier", "ecutwfc_ry", "ecutrho_ry",
        "kpoint_spacing_inv_angstrom",
    ):
        _same_float(provisional.get(key), selected.get(key), f"selected_settings.{key}")
    if (
        sweep_summary.get("qe_program_version") != payload.get("qe_program_version")
        or sweep_summary.get("pw_executable_sha256")
        != payload.get("pw_executable_sha256")
    ):
        raise ValueError("Certificate provisional QE executable lineage mismatch")
    require_same_execution_provenance(
        sweep_summary.get("execution_provenance"),
        payload.get("execution_provenance"),
        label="certificate payload",
    )
    sweep_points = sweep_summary.get("points")
    if (
        not isinstance(sweep_points, list)
        or not sweep_points
        or sweep_summary.get("point_count") != len(sweep_points)
    ):
        raise ValueError("Certificate provisional sweep point inventory is invalid")
    sweep_csv_rows = _read_csv(Path(artifacts["sweep_results"]["path"]))
    sweep_representatives = sweep_preflight.get("representatives")
    if not isinstance(sweep_representatives, list) or not sweep_representatives:
        raise ValueError("Certificate sweep representative inventory is missing")
    verified_sweep_evidence = verify_provisional_sweep_evidence(
        summary=sweep_summary,
        csv_rows=sweep_csv_rows,
        locked_study_points=sweep_preflight.get("study_points") or [],
        representative_ids=[
            str(record.get("representative_id") or "")
            for record in sweep_representatives
        ],
        expected_protocol=sweep_preflight.get("protocol_values"),
    )
    for key, expected in verified_sweep_evidence["global_selection"].items():
        _same_float(
            expected, selected.get(key), f"selected_settings.{key}"
        )
    expected_production = derive_strictest_tested_settings(
        protocol=sweep_preflight.get("protocol_values"),
        base_ecutwfc_ry=sweep_preflight.get("base_ecutwfc_ry"),
        base_ecutrho_ry=sweep_preflight.get("base_ecutrho_ry"),
    )
    production = payload.get("production_settings")
    if not isinstance(production, dict) or set(production) != set(expected_production):
        raise ValueError("Certificate production settings are missing or invalid")
    for key, expected in expected_production.items():
        _same_float(expected, production.get(key), f"production_settings.{key}")
    for representative_id, anchor in verified_sweep_evidence[
        "anchors_by_representative"
    ].items():
        for key, point_key in (
            ("cutoff_pair_multiplier", "cutoff_pair_multiplier"),
            ("ecutwfc_ry", "ecutwfc_ry"),
            ("ecutrho_ry", "ecutrho_ry"),
            (
                "kpoint_spacing_inv_angstrom",
                "requested_kpoint_spacing_inv_angstrom",
            ),
        ):
            _same_float(
                expected_production[key], anchor.get(point_key),
                f"strictest anchor:{representative_id}:{point_key}",
            )
    sweep_csv_by_id = {row.get("point_id", ""): row for row in sweep_csv_rows}
    if len(sweep_csv_by_id) != len(sweep_csv_rows) or len(sweep_csv_rows) != len(sweep_points):
        raise ValueError("Certificate provisional sweep CSV inventory is invalid")
    for point in sweep_points:
        point_id = str((point or {}).get("point_id") or "") if isinstance(point, dict) else ""
        row = sweep_csv_by_id.get(point_id)
        if not point_id or row is None:
            raise ValueError("Certificate provisional sweep point is missing")
        try:
            point_execution = json.loads(
                str(point.get("execution_provenance") or "")
            )
            row_execution = json.loads(
                str(row.get("execution_provenance") or "")
            )
            require_same_execution_provenance(
                point_execution,
                payload.get("execution_provenance"),
                label="certificate payload",
            )
            require_same_execution_provenance(
                row_execution,
                point_execution,
                label="provisional sweep summary point",
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                "Certificate provisional sweep execution lineage mismatch"
            ) from exc

    if (
        confirmation_preflight.get("workflow_version")
        != "qe_pbe_convergence_confirmation_v1"
        or confirmation_preflight.get("stage") != "confirmation"
        or confirmation_preflight.get("calculation") != "scf"
        or confirmation_preflight.get("coverage_complete") is not True
        or confirmation_preflight.get("certificate_eligible") is not False
    ):
        raise ValueError("Certificate confirmation preflight state is invalid")
    confirmation_settings_payload = confirmation_preflight.get(
        "convergence_settings_payload"
    )
    confirmation_settings_hash = str(
        confirmation_preflight.get("convergence_settings_hash") or ""
    ).lower()
    if (
        not isinstance(confirmation_settings_payload, dict)
        or canonical_digest(confirmation_settings_payload) != confirmation_settings_hash
        or confirmation_settings_hash != payload.get("confirmation_settings_hash")
        or confirmation_preflight.get("source_sweep_settings_hash")
        != sweep_settings_hash
    ):
        raise ValueError("Certificate confirmation settings lineage is invalid")
    if (
        confirmation_preflight.get("sweep_preflight_sha256")
        != artifacts["sweep_preflight"]["sha256"]
        or confirmation_preflight.get("provisional_summary_sha256")
        != artifacts["sweep_summary"]["sha256"]
        or confirmation_preflight.get("provisional_results_sha256")
        != artifacts["sweep_results"]["sha256"]
        or confirmation_preflight.get("queue_manifest_sha256")
        != artifacts["confirmation_queue"]["sha256"]
    ):
        raise ValueError("Certificate confirmation artifact lineage mismatch")
    for key, artifact_label, hash_key in (
        ("config", "config", "config_sha256"),
        ("pseudo_manifest", "pseudo_manifest", "pseudo_manifest_sha256"),
        ("protocol", "protocol", "protocol_sha256"),
    ):
        if (
            Path(str(confirmation_preflight.get(key) or "")).resolve()
            != Path(artifacts[artifact_label]["path"])
            or confirmation_preflight.get(hash_key)
            != artifacts[artifact_label]["sha256"]
        ):
            raise ValueError(f"Certificate confirmation {key} lineage mismatch")
    if (
        confirmation_preflight.get("required_elements") != payload.get("required_elements")
        or confirmation_preflight.get("covered_elements") != payload.get("covered_elements")
        or confirmation_preflight.get("selected_global_settings")
        != payload.get("selected_settings")
        or confirmation_preflight.get("baseline_qe_program_version")
        != payload.get("qe_program_version")
        or confirmation_preflight.get("baseline_pw_executable_sha256")
        != payload.get("pw_executable_sha256")
    ):
        raise ValueError("Certificate confirmation scientific lineage mismatch")
    require_same_execution_provenance(
        confirmation_preflight.get("baseline_execution_provenance"),
        payload.get("execution_provenance"),
        label="certificate payload",
    )

    required_confirmation_state = {
        "collector_version": "qe_convergence_confirmation_collector_v1",
        "status": "confirmation_passed",
        "certificate_eligible": True,
        "confirmation_required": False,
        "coverage_complete": True,
    }
    if any(
        confirmation_summary.get(key) != value
        for key, value in required_confirmation_state.items()
    ):
        raise ValueError("Certificate confirmation summary did not pass")
    if (
        confirmation_summary.get("confirmation_settings_hash")
        != confirmation_settings_hash
        or confirmation_summary.get("source_sweep_settings_hash")
        != sweep_settings_hash
        or confirmation_summary.get("confirmation_preflight_sha256")
        != artifacts["confirmation_preflight"]["sha256"]
        or confirmation_summary.get("queue_manifest_sha256")
        != artifacts["confirmation_queue"]["sha256"]
        or confirmation_summary.get("results_sha256")
        != artifacts["confirmation_results"]["sha256"]
        or confirmation_summary.get("qe_program_version")
        != payload.get("qe_program_version")
        or confirmation_summary.get("pw_executable_sha256")
        != payload.get("pw_executable_sha256")
        or confirmation_summary.get("selected_settings")
        != payload.get("selected_settings")
    ):
        raise ValueError("Certificate confirmation summary lineage mismatch")
    require_same_execution_provenance(
        confirmation_summary.get("execution_provenance"),
        payload.get("execution_provenance"),
        label="certificate payload",
    )
    points = confirmation_summary.get("points")
    if (
        not isinstance(points, list)
        or not points
        or confirmation_summary.get("point_count") != len(points)
        or confirmation_summary.get("gate_status_counts")
        != {"confirmation_passed": len(points)}
    ):
        raise ValueError("Certificate confirmation point count/status is invalid")
    csv_rows = _read_csv(Path(artifacts["confirmation_results"]["path"]))
    csv_by_id = {row.get("point_id", ""): row for row in csv_rows}
    if len(csv_by_id) != len(csv_rows):
        raise ValueError("Certificate confirmation CSV point IDs are duplicated")
    protocol = sweep_summary.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("Certificate convergence protocol values are missing")
    tolerances = {
        "energy_delta_to_anchor_mev_per_atom": float(
            protocol["energy_tolerance_mev_per_atom"]
        ),
        "max_force_delta_to_anchor_ev_per_angstrom": float(
            protocol["force_component_tolerance_ev_per_angstrom"]
        ),
        "max_stress_delta_to_anchor_kbar": float(
            protocol["stress_component_tolerance_kbar"]
        ),
    }
    passed_by_rep: dict[str, dict[str, Any]] = {}
    for point in points:
        point_id = str((point or {}).get("point_id") or "") if isinstance(point, dict) else ""
        rep_id = str((point or {}).get("representative_id") or "") if isinstance(point, dict) else ""
        row = csv_by_id.get(point_id)
        if (
            not point_id or not rep_id or rep_id in passed_by_rep or row is None
            or point.get("confirmation_gate_status") != "confirmation_passed"
            or point.get("point_within_tolerances") is not True
            or json.loads(str(point.get("gate_failures") or "[]")) != []
            or row.get("confirmation_gate_status") != "confirmation_passed"
            or row.get("point_within_tolerances") != "True"
        ):
            raise ValueError("Certificate confirmation point did not pass")
        for field, tolerance in tolerances.items():
            value = float(point[field])
            if not math.isfinite(value) or value < 0 or value > tolerance:
                raise ValueError(f"Certificate confirmation tolerance failed: {field}")
            if not math.isclose(value, float(row[field]), rel_tol=1e-12, abs_tol=1e-10):
                raise ValueError(f"Certificate confirmation CSV mismatch: {field}")
        if (
            point.get("qe_program_version") != payload.get("qe_program_version")
            or point.get("pw_executable_sha256") != payload.get("pw_executable_sha256")
        ):
            raise ValueError("Certificate confirmation point QE lineage mismatch")
        try:
            point_execution = json.loads(
                str(point.get("execution_provenance") or "")
            )
            row_execution = json.loads(
                str(row.get("execution_provenance") or "")
            )
            require_same_execution_provenance(
                point_execution,
                payload.get("execution_provenance"),
                label="certificate payload",
            )
            require_same_execution_provenance(
                row_execution,
                point_execution,
                label="confirmation summary point",
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                "Certificate confirmation point execution lineage mismatch"
            ) from exc
        passed_by_rep[rep_id] = point
    certificate_reps = payload.get("representatives") or []
    if {record.get("representative_id") for record in certificate_reps} != set(passed_by_rep):
        raise ValueError("Certificate representative inventory differs from confirmation")
    for record in certificate_reps:
        point = passed_by_rep[record["representative_id"]]
        if record.get("point_id") != point.get("point_id") or record.get("status") != "passed":
            raise ValueError("Certificate representative point lineage mismatch")
        for field in tolerances:
            if not math.isclose(
                float(record[field]), float(point[field]), rel_tol=1e-12, abs_tol=1e-10
            ):
                raise ValueError("Certificate representative metric lineage mismatch")


def verify_convergence_certificate(
    certificate_path: Path,
    *,
    required_elements: list[str] | set[str] | tuple[str, ...] | None = None,
    config_path: Path | None = None,
    pseudo_manifest_path: Path | None = None,
    pw_executable_path: Path | None = None,
) -> dict[str, Any]:
    """Verify a certificate and every artifact it locks.

    ``required_elements`` may be a subset of the certified campaign (for
    example a competing reference phase).  A superset is rejected.  The
    executable check is optional during plan-only preparation, but the runner
    supplies the resolved executable and enforces it immediately before launch.
    """
    certificate_path = Path(certificate_path).expanduser().resolve()
    certificate = _load_json(certificate_path)
    if certificate.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Expected certificate schema {SCHEMA_VERSION}")
    if certificate.get("status") != "passed_tested_window":
        raise ValueError("Convergence certificate did not pass the tested window")
    if certificate.get("certificate_eligible") is not True:
        raise ValueError("Convergence certificate is not eligible for production")
    if certificate.get("confirmation_required") is not False:
        raise ValueError("Convergence confirmation is still required")
    payload = certificate.get("certificate_payload")
    if not isinstance(payload, dict):
        raise ValueError("Convergence certificate payload is missing")
    recorded_digest = str(
        certificate.get("certificate_payload_sha256") or ""
    ).strip().lower()
    if not HASH_RE.fullmatch(recorded_digest) or canonical_digest(payload) != recorded_digest:
        raise ValueError("Convergence certificate canonical payload hash mismatch")
    if certificate.get("certificate_id") != f"qeconv-{recorded_digest[:20]}":
        raise ValueError("Convergence certificate ID does not match its payload")
    if payload.get("target_profile") != "finite_smearing_static_scf":
        raise ValueError("Unsupported convergence certificate target profile")
    if payload.get("coverage_complete") is not True:
        raise ValueError("Convergence certificate element coverage is incomplete")

    certified_required = payload.get("required_elements")
    certified_covered = payload.get("covered_elements")
    if not isinstance(certified_required, list) or not isinstance(certified_covered, list):
        raise ValueError("Certificate element inventories are missing")
    if (
        any(not isinstance(value, str) or not value for value in certified_required)
        or any(not isinstance(value, str) or not value for value in certified_covered)
        or len(certified_required) != len(set(certified_required))
        or len(certified_covered) != len(set(certified_covered))
        or not set(certified_required).issubset(certified_covered)
    ):
        raise ValueError("Certificate element inventories are invalid")
    requested = set(required_elements or [])
    unseen = sorted(requested - set(certified_covered))
    if unseen:
        raise ValueError(
            "Required elements are outside convergence-certificate coverage: "
            + ",".join(unseen)
        )

    selected = payload.get("selected_settings")
    if not isinstance(selected, dict):
        raise ValueError("Certificate selected settings are missing")
    selected_values = {
        "cutoff_pair_multiplier": _positive_float(
            selected.get("cutoff_pair_multiplier"), "cutoff_pair_multiplier"
        ),
        "ecutwfc_ry": _positive_float(selected.get("ecutwfc_ry"), "ecutwfc_ry"),
        "ecutrho_ry": _positive_float(selected.get("ecutrho_ry"), "ecutrho_ry"),
        "kpoint_spacing_inv_angstrom": _positive_float(
            selected.get("kpoint_spacing_inv_angstrom"),
            "kpoint_spacing_inv_angstrom",
        ),
    }
    if selected_values["ecutrho_ry"] < selected_values["ecutwfc_ry"]:
        raise ValueError("Certificate ecutrho_ry is below ecutwfc_ry")
    production = payload.get("production_settings")
    if not isinstance(production, dict):
        raise ValueError("Certificate production settings are missing")
    production_values = {
        "cutoff_pair_multiplier": _positive_float(
            production.get("cutoff_pair_multiplier"),
            "production cutoff_pair_multiplier",
        ),
        "ecutwfc_ry": _positive_float(
            production.get("ecutwfc_ry"), "production ecutwfc_ry"
        ),
        "ecutrho_ry": _positive_float(
            production.get("ecutrho_ry"), "production ecutrho_ry"
        ),
        "kpoint_spacing_inv_angstrom": _positive_float(
            production.get("kpoint_spacing_inv_angstrom"),
            "production kpoint_spacing_inv_angstrom",
        ),
    }
    if set(production) != set(production_values):
        raise ValueError("Certificate production settings contain unknown fields")
    if (
        production_values["ecutwfc_ry"] + 1e-10 < selected_values["ecutwfc_ry"]
        or production_values["ecutrho_ry"] + 1e-8 < selected_values["ecutrho_ry"]
        or production_values["kpoint_spacing_inv_angstrom"]
        > selected_values["kpoint_spacing_inv_angstrom"] + 1e-10
    ):
        raise ValueError(
            "Certificate production settings are less conservative than the "
            "confirmed selection"
        )
    if payload.get("production_settings_scope") != PRODUCTION_TRANSFER_SCOPE:
        raise ValueError("Certificate production transfer limitation is missing")
    base_wfc = _positive_float(payload.get("base_ecutwfc_ry"), "base_ecutwfc_ry")
    base_rho = _positive_float(payload.get("base_ecutrho_ry"), "base_ecutrho_ry")
    if (
        selected_values["ecutwfc_ry"] + 1e-10 < base_wfc
        or selected_values["ecutrho_ry"] + 1e-8 < base_rho
    ):
        raise ValueError("Certificate cutoffs are below verified SSSP base cutoffs")

    settings_hash = str(payload.get("source_sweep_settings_hash") or "").lower()
    qe_version = str(payload.get("qe_program_version") or "").strip()
    executable_sha = str(payload.get("pw_executable_sha256") or "").lower()
    execution_provenance = validate_execution_provenance(
        payload.get("execution_provenance")
    )
    if not HASH_RE.fullmatch(settings_hash):
        raise ValueError("Certificate source sweep settings hash is invalid")
    if not qe_version:
        raise ValueError("Certificate QE program version is missing")
    if not HASH_RE.fullmatch(executable_sha):
        raise ValueError("Certificate QE executable hash is invalid")

    artifacts = payload.get("artifacts")
    required_artifacts = (
        "sweep_preflight", "sweep_results", "sweep_summary",
        "confirmation_preflight", "confirmation_queue",
        "confirmation_results", "confirmation_summary",
        "config", "pseudo_manifest", "protocol",
    )
    if not isinstance(artifacts, dict):
        raise ValueError("Certificate locked-artifact inventory is missing")
    verified_artifacts = {
        label: _verify_locked_artifact(artifacts.get(label), label)
        for label in required_artifacts
    }

    if config_path is not None:
        resolved = Path(config_path).expanduser().resolve()
        if (
            not resolved.is_file()
            or sha256_file(resolved) != verified_artifacts["config"]["sha256"]
        ):
            raise ValueError("Production config does not match the convergence certificate")
    if pseudo_manifest_path is not None:
        resolved = Path(pseudo_manifest_path).expanduser().resolve()
        if (
            not resolved.is_file()
            or sha256_file(resolved)
            != verified_artifacts["pseudo_manifest"]["sha256"]
        ):
            raise ValueError(
                "Production pseudopotential manifest does not match the convergence certificate"
            )

    pseudo_records = payload.get("pseudopotentials")
    if not isinstance(pseudo_records, list) or not pseudo_records:
        raise ValueError("Certificate pseudopotential inventory is missing")
    pseudo_elements: set[str] = set()
    for index, record in enumerate(pseudo_records):
        element = str((record or {}).get("element") or "") if isinstance(record, dict) else ""
        if not element or element in pseudo_elements:
            raise ValueError("Certificate pseudopotential element inventory is invalid")
        pseudo_elements.add(element)
        _verify_locked_artifact(record, f"pseudopotential:{element or index}")
    if pseudo_elements != set(certified_covered):
        raise ValueError("Certificate pseudopotential coverage differs from element coverage")

    representatives = payload.get("representatives")
    if not isinstance(representatives, list) or not representatives:
        raise ValueError("Certificate confirmation representatives are missing")
    rep_ids: set[str] = set()
    for record in representatives:
        if not isinstance(record, dict):
            raise ValueError("Certificate representative record is invalid")
        rep_id = str(record.get("representative_id") or "")
        if not rep_id or rep_id in rep_ids or record.get("status") != "passed":
            raise ValueError("Certificate representative confirmation is invalid")
        rep_ids.add(rep_id)

    _verify_semantic_lineage(payload=payload, artifacts=verified_artifacts)

    if pw_executable_path is not None:
        executable = Path(pw_executable_path).expanduser().resolve()
        if not executable.is_file() or sha256_file(executable) != executable_sha:
            raise ValueError(
                "Resolved pw.x does not match the executable used for convergence"
            )

    return {
        "certificate_path": str(certificate_path),
        "certificate_sha256": sha256_file(certificate_path),
        "certificate_id": certificate["certificate_id"],
        "certificate_payload_sha256": recorded_digest,
        "payload": payload,
        "selected_settings": selected_values,
        "production_settings": production_values,
        "production_settings_scope": str(
            payload.get("production_settings_scope") or ""
        ),
        "qe_program_version": qe_version,
        "pw_executable_sha256": executable_sha,
        "execution_provenance": execution_provenance,
        "covered_elements": sorted(certified_covered),
        "verified_artifacts": verified_artifacts,
    }
