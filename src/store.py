"""Append-only experiment store and the experiment budget.

TRUSTED INFRASTRUCTURE. JSONL is the source of truth: one line per experiment,
fsync'd on append, so a killed run loses at most the in-flight experiment.
Prompts, responses and artifacts are written as separate files and referenced by
path, to keep the JSONL readable.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .utils import ensure_dir, _json_default


class BudgetExhausted(RuntimeError):
    pass


@dataclass
class Budget:
    """Hard cap on experiments. Failed and invalid experiments consume budget --
    otherwise an agent that emits garbage would get unlimited free retries and
    any cross-method comparison would be meaningless."""
    max_experiments: int
    used: int = 0

    def remaining(self) -> int:
        return max(self.max_experiments - self.used, 0)

    def check(self) -> None:
        if self.remaining() <= 0:
            raise BudgetExhausted(f"experiment budget exhausted ({self.max_experiments})")

    def consume(self) -> None:
        self.used += 1

    def as_dict(self) -> dict:
        return {"max_experiments": self.max_experiments, "used": self.used,
                "remaining": self.remaining()}


@dataclass
class ExperimentStore:
    path: str | Path
    run_id: str = ""
    _records: list[dict] = field(default_factory=list)

    def __post_init__(self):
        self.path = Path(self.path)
        ensure_dir(self.path.parent)

    def append(self, record: dict) -> None:
        self._records.append(record)
        with open(self.path, "a") as fh:
            fh.write(json.dumps(record, default=_json_default) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    # ----- reads ---------------------------------------------------------- #
    def records_this_run(self) -> list[dict]:
        return list(self._records)

    def iter_all(self) -> Iterator[dict]:
        if not Path(self.path).exists():
            return
        with open(self.path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)

    def best_this_run(self, metric: str = "auroc") -> dict | None:
        return best_of(self._records, metric)


def best_of(records: list[dict], metric: str = "auroc") -> dict | None:
    scored = [
        r for r in records
        if r.get("status") == "success" and (r.get("val_metrics") or {}).get(metric) is not None
    ]
    if not scored:
        return None
    return max(scored, key=lambda r: r["val_metrics"][metric])


def leaderboard(records: list[dict], metric: str = "auroc") -> list[dict]:
    rows = []
    for r in records:
        vm = r.get("val_metrics") or {}
        tm = r.get("train_metrics") or {}
        rows.append({
            "run_id": r.get("run_id"),
            "experiment_id": r.get("experiment_id"),
            "parent": r.get("parent_experiment"),
            "status": r.get("status"),
            "family": ((r.get("proposal") or {}).get("model") or {}).get("family"),
            "n_features": r.get("n_features"),
            f"train_{metric}": tm.get(metric),
            f"val_{metric}": vm.get(metric),
            "val_auprc": vm.get("auprc"),
            "val_f1": vm.get("f1"),
            "runtime_s": r.get("runtime_seconds"),
        })
    rows.sort(key=lambda d: (d[f"val_{metric}"] is None, -(d[f"val_{metric}"] or 0)))
    return rows


def format_leaderboard(rows: list[dict], limit: int = 20) -> str:
    if not rows:
        return "(no experiments)"
    cols = ["experiment_id", "status", "family", "n_features",
            "train_auroc", "val_auroc", "val_auprc", "val_f1", "runtime_s"]
    widths = {c: max(len(c), *(len(_fmt(r.get(c))) for r in rows[:limit])) for c in cols}
    head = "  ".join(c.ljust(widths[c]) for c in cols)
    lines = [head, "-" * len(head)]
    for r in rows[:limit]:
        lines.append("  ".join(_fmt(r.get(c)).ljust(widths[c]) for c in cols))
    return "\n".join(lines)


def _fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)
