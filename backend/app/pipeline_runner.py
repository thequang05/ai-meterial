"""End-to-end pipeline runner.

Executes research/phase_2/run_pipeline.py as a subprocess so the FE can
trigger the full Stage 0 -> Stage 4 pipeline (or any subset) over HTTP
without us re-implementing the steps in Python.

Outputs are kept in ``backend/runs/<run_id>/`` so the backend can serve
them back to the UI, and so the user can audit a previous run on disk.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# Layout: backend/app/pipeline_runner.py
#   parents[0] = backend/app, [1] = backend, [2] = ai-meterial  (project root)
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _BACKEND_ROOT.parent
_PHASE2 = _PROJECT_ROOT / "research" / "phase_2"
_RUNS_DIR = _BACKEND_ROOT / "runs"


@dataclass
class _RunState:
    id: str
    prompt: str
    created_at: str
    status: str = "running"
    return_code: int | None = None
    error: str | None = None
    log: str = ""
    parsed_args: dict = field(default_factory=dict)
    method: str | None = None
    llm_model: str | None = None
    stage0_candidates: list[dict] = field(default_factory=list)
    ranked: list[dict] = field(default_factory=list)
    duration_s: float | None = None
    output_dir: str = ""


_RUNS: dict[str, _RunState] = {}
_RUNS_LOCK = threading.Lock()


def list_runs() -> list[dict]:
    with _RUNS_LOCK:
        items = []
        for r in _RUNS.values():
            d = asdict(r)
            d.pop("log", None)
            d["log_excerpt"] = _excerpt(d.pop("log", "") or "", 600)
            # re-key created_at as already ISO string
            items.append(d)
        items.sort(key=lambda r: r["created_at"], reverse=True)
        return items


def get_run(run_id: str) -> _RunState | None:
    with _RUNS_LOCK:
        r = _RUNS.get(run_id)
        return r


def start_run(
    *,
    prompt: str,
    limit: int,
    model: str | None,
    use_llm: bool,
    skip_stage4: bool,
) -> _RunState:
    run_id = uuid.uuid4().hex[:12]
    out_dir = _RUNS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    state = _RunState(
        id=run_id,
        prompt=prompt,
        created_at=datetime.now(timezone.utc).isoformat(),
        output_dir=str(out_dir),
    )
    with _RUNS_LOCK:
        _RUNS[run_id] = state

    # Run inline so the HTTP request blocks until the pipeline finishes.
    # The whole 5-stage Phase 2 ML pipeline completes in ~5-10s, well
    # within FastAPI's default timeouts. If we ever need true async,
    # we can swap this for a BackgroundTasks approach without changing
    # the response shape.
    _execute(state, prompt, limit, model, use_llm, skip_stage4)
    return state


# ──────────────────────────────────────────────────────────────────────
# Execution
# ──────────────────────────────────────────────────────────────────────
def _execute(
    state: _RunState,
    prompt: str,
    limit: int,
    model: str | None,
    use_llm: bool,
    skip_stage4: bool,
) -> None:
    started = time.monotonic()
    out_dir = Path(state.output_dir)

    # Compose env. Carry LM Studio settings if present, else use defaults
    # already baked into nl_to_candidates.py.
    env = os.environ.copy()
    if model:
        env["LMSTUDIO_MODEL"] = model

    cmd: list[str] = [
        sys.executable,
        str(_PHASE2 / "run_pipeline.py"),
        "--prompt", prompt,
        "--output-dir", str(out_dir),
    ]
    if not use_llm:
        cmd.append("--no-llm")
    if model and use_llm:
        cmd.extend(["--model", model])
    if skip_stage4:
        cmd.append("--skip-stage4")

    state.log += f"$ {' '.join(cmd)}\n"
    proc = subprocess.Popen(
        cmd,
        cwd=str(_PHASE2),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        state.log += line
        # keep log bounded to avoid OOM if a user runs something huge
        if len(state.log) > 200_000:
            state.log = state.log[-200_000:]
    rc = proc.wait()
    state.return_code = rc
    state.duration_s = round(time.monotonic() - started, 3)

    if rc != 0:
        state.status = "failed"
        state.error = f"pipeline exited with rc={rc}"
        return

    # parse stage0 + stage4 outputs
    try:
        stage0_json = out_dir / "stage0_candidates.json"
        if stage0_json.exists():
            data = json.loads(stage0_json.read_text())
            state.parsed_args = data.get("parsed_args") or {}
            state.method = data.get("method")
            state.llm_model = data.get("llm_model") or model
            state.stage0_candidates = data.get("candidates") or []
            # apply UI limit cap to candidates we expose
            state.stage0_candidates = state.stage0_candidates[:limit]
    except Exception as e:
        state.log += f"[parse-stage0] {e}\n"

    csv_path = out_dir / "ml_thermal_evaluation.csv"
    if csv_path.exists():
        try:
            with csv_path.open() as f:
                reader = csv.DictReader(f)
                for row in reader:
                    state.ranked.append({
                        "rank": int(row.get("ml_rank") or row.get("rank") or 0),
                        "candidate_id": row.get("candidate_id", ""),
                        "formula": row.get("formula", ""),
                        "gnn_ef_post": _to_float(row.get("gnn_formation_energy_post_chgnet_ev_per_atom")),
                        "mp_best_formula": row.get("mp_best_formula") or None,
                        "mp_best_formation_energy_per_atom": _to_float(row.get("mp_best_formation_energy_per_atom")),
                        "ml_estimated_hull_gap_ev_per_atom": _to_float(row.get("ml_estimated_hull_gap_ev_per_atom")),
                        "status": row.get("thermal_proxy_status") or None,
                        "cif_path": row.get("cif_path") or None,
                    })
        except Exception as e:
            state.log += f"[parse-stage4] {e}\n"

    state.status = "succeeded"


def _to_float(v: Any) -> float | None:
    if v in (None, "", "None"):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _excerpt(text: str, n: int) -> str:
    if len(text) <= n:
        return text
    return "...\n" + text[-n:]


def summary(state: _RunState) -> dict:
    d = asdict(state)
    d.pop("log", None)
    d["log_excerpt"] = _excerpt(state.log, 1200)
    d["candidate_count"] = len(state.stage0_candidates)
    d["ranked_count"] = len(state.ranked)
    return d