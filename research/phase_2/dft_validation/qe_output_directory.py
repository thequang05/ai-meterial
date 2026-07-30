"""Fail-closed output-directory handling for QE post-processing stages."""

from __future__ import annotations

from pathlib import Path


def require_fresh_output_dir(output_dir: Path) -> Path:
    """Return an absolute output path only when it is absent or empty.

    Collectors are intentionally append-incompatible: mixing artifacts from
    separate executions can make a stale result look like current evidence.
    This check does not create the directory, so callers can validate all
    inputs before their first write.
    """

    resolved = Path(output_dir).expanduser().resolve()
    if resolved.exists() and not resolved.is_dir():
        raise FileExistsError(f"QE output path is not a directory: {resolved}")
    if resolved.exists():
        existing = sorted(entry.name for entry in resolved.iterdir())
        if existing:
            raise FileExistsError(
                "Refusing to collect QE results in a non-empty output "
                f"directory: {resolved}; existing={existing}"
            )
    return resolved
