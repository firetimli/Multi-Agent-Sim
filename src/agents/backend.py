"""LLM backends. Codex CLI is the agent backbone; mock is for offline smoke tests.

This file is TRUSTED, human-written code -- it is the plumbing that elicits agent
output. The *agent-controlled* artifacts are the JSON proposals/critiques it
returns, which only ever reach disk under generated/ and runs/.

The backbone is deliberately a swappable interface: the paper's claim is about
the multi-agent workflow, not about a particular model, so the model must be
replaceable without touching the orchestrator.
"""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from typing import Any


class BackendError(RuntimeError):
    pass


@dataclass
class Backend:
    name: str = "base"
    calls: int = 0

    def complete(self, prompt: str, kind: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Codex CLI
# --------------------------------------------------------------------------- #
@dataclass
class CodexCLIBackend(Backend):
    """Runs the Codex CLI as a subprocess, prompt on stdin, completion on stdout.

    `cmd` is config-driven because CLI flags change; set agent.codex_cmd in the
    dataset config to match your installed version. The CLI is invoked in a
    scratch cwd so it has no reason to touch the repository.
    """
    cmd: list[str] = field(default_factory=lambda: ["codex", "exec", "--skip-git-repo-check", "-"])
    cwd: str | None = None
    timeout: int = 600
    name: str = "codex_cli"

    def complete(self, prompt: str, kind: str) -> str:
        self.calls += 1
        t0 = time.time()
        try:
            proc = subprocess.run(
                self.cmd, input=prompt, capture_output=True, text=True,
                timeout=self.timeout, cwd=self.cwd,
            )
        except FileNotFoundError as exc:
            raise BackendError(
                f"could not launch {self.cmd[0]!r}. Install the Codex CLI or run with "
                f"--backend mock."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise BackendError(f"{self.name} timed out after {self.timeout}s") from exc
        if proc.returncode != 0:
            raise BackendError(
                f"{self.name} exited {proc.returncode}: {(proc.stderr or '')[-800:]}"
            )
        out = proc.stdout or ""
        if not out.strip():
            raise BackendError(f"{self.name} returned empty output ({time.time() - t0:.1f}s)")
        return out


# --------------------------------------------------------------------------- #
# Mock (deterministic, offline)
# --------------------------------------------------------------------------- #
_MOCK_PROPOSALS: list[dict] = [
    {
        "rationale": "Start simple: cheap linear baseline on volume and recency features.",
        "features": [
            {"name": "history_count", "parameters": {}},
            {"name": "time_since_last_event", "parameters": {}},
            {"name": "event_type_counts", "parameters": {"normalize": False}},
        ],
        "model": {"family": "logistic_regression",
                  "hyperparameters": {"C": 1.0, "class_weight": "balanced"}},
    },
    {
        "rationale": "Move to gradient boosting and add diversity/intensity features.",
        "features": [
            {"name": "history_count", "parameters": {}},
            {"name": "unique_item_count", "parameters": {}},
            {"name": "time_since_last_event", "parameters": {}},
            {"name": "recent_event_count", "parameters": {"window_days": 1.0}},
            {"name": "event_type_counts", "parameters": {"normalize": False}},
        ],
        "model": {"family": "lightgbm",
                  "hyperparameters": {"learning_rate": 0.05, "num_leaves": 31}},
    },
    {
        "rationale": "Regularise the booster and add temporal-rhythm features.",
        "features": [
            {"name": "history_count", "parameters": {}},
            {"name": "unique_item_count", "parameters": {}},
            {"name": "time_since_last_event", "parameters": {}},
            {"name": "mean_inter_event_time", "parameters": {}},
            {"name": "recent_event_count", "parameters": {"window_days": 3.0}},
            {"name": "item_repeat_ratio", "parameters": {}},
            {"name": "session_length", "parameters": {"gap_minutes": 30.0}},
            {"name": "event_type_counts", "parameters": {"normalize": True}},
        ],
        "model": {"family": "lightgbm",
                  "hyperparameters": {"learning_rate": 0.03, "num_leaves": 15,
                                      "min_child_samples": 100, "reg_lambda": 5.0,
                                      "colsample_bytree": 0.8, "subsample": 0.8}},
    },
    {
        "rationale": "Try a different inductive bias with the same feature set.",
        "features": [
            {"name": "history_count", "parameters": {}},
            {"name": "unique_item_count", "parameters": {}},
            {"name": "time_since_last_event", "parameters": {}},
            {"name": "recent_event_count", "parameters": {"window_days": 2.0}},
            {"name": "session_length", "parameters": {"gap_minutes": 45.0}},
            {"name": "event_type_counts", "parameters": {"normalize": False}},
        ],
        "model": {"family": "random_forest",
                  "hyperparameters": {"n_estimators": 300, "min_samples_leaf": 20,
                                      "class_weight": "balanced_subsample"}},
    },
]

_MOCK_CRITIQUES: list[dict] = [
    {"diagnosis": ["underfitting"],
     "evidence": ["train AUROC and validation AUROC are both low and close together"],
     "recommendations": ["use a non-linear model family", "add more behavioural features"]},
    {"diagnosis": ["overfitting"],
     "evidence": ["train AUROC is much higher than validation AUROC"],
     "recommendations": ["reduce model complexity", "increase regularisation"]},
    {"diagnosis": ["class_imbalance"],
     "evidence": ["positive rate is low and F1 is far below AUROC"],
     "recommendations": ["adjust class weighting", "keep AUROC as the selection metric"]},
]


@dataclass
class MockBackend(Backend):
    """Deterministic canned responses. Exercises the full loop with no API calls."""
    name: str = "mock"
    _prop_i: int = 0
    _crit_i: int = 0

    def complete(self, prompt: str, kind: str) -> str:
        from ..utils import jdump
        self.calls += 1
        if kind == "critic":
            obj = _MOCK_CRITIQUES[self._crit_i % len(_MOCK_CRITIQUES)]
            self._crit_i += 1
            return jdump(obj)
        i = self._prop_i
        self._prop_i += 1
        base = dict(_MOCK_PROPOSALS[i % len(_MOCK_PROPOSALS)])
        if i >= len(_MOCK_PROPOSALS):  # deterministic variation once the list is used up
            hp = dict(base["model"]["hyperparameters"])
            if base["model"]["family"] == "lightgbm":
                hp["learning_rate"] = round(0.02 + 0.01 * (i % 4), 3)
                hp["num_leaves"] = 8 * (1 + i % 5)
            base["model"] = {"family": base["model"]["family"], "hyperparameters": hp}
            base["rationale"] = f"Mock variation {i}: perturb hyperparameters."
        return jdump(base)


def make_backend(agent_cfg: dict, override: str | None = None) -> Backend:
    kind = (override or agent_cfg.get("backend", "mock")).lower()
    if kind == "mock":
        return MockBackend()
    if kind in ("codex", "codex_cli"):
        return CodexCLIBackend(
            cmd=list(agent_cfg.get("codex_cmd",
                                   ["codex", "exec", "--skip-git-repo-check", "-"])),
            cwd=agent_cfg.get("codex_cwd"),
            timeout=int(agent_cfg.get("timeout_seconds", 600)),
        )
    raise ValueError(f"unknown agent backend {kind!r} (use 'codex_cli' or 'mock')")
