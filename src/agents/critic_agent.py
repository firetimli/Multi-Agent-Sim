"""Critic Agent: diagnoses one experiment and recommends what to change next.

Sees the proposal, train metrics, validation metrics, the previous best, and
recent history. It never sees test data -- there is none in Version 0.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..utils import ProposalError, extract_json, jdump, read_text, render
from .backend import Backend, BackendError
from .modeling_agent import AgentResult

ALLOWED_DIAGNOSES = (
    "underfitting", "overfitting", "class_imbalance", "weak_features",
    "optimization_unstable", "runtime_bound", "leakage_suspected",
    "insufficient_signal", "ok",
)
MAX_ITEMS = 6


class CriticAgent:
    kind = "critic"

    def __init__(self, backend: Backend, prompt_path: str | Path, max_repair_attempts: int = 1):
        self.backend = backend
        self.template = read_text(prompt_path)
        self.max_repair_attempts = max_repair_attempts

    def build_prompt(self, context: dict, repair_feedback: str | None = None) -> str:
        return render(self.template, {
            "DATASET_PROFILE": jdump(context["dataset_profile"]),
            "PROPOSAL": jdump(context["proposal"]),
            "TRAIN_METRICS": jdump(context["train_metrics"]),
            "VAL_METRICS": jdump(context["val_metrics"]),
            "OVERFIT_GAP": jdump(context["overfit_gap"]),
            "RUN_INFO": jdump(context["run_info"]),
            "BEST_SO_FAR": jdump(context["best_so_far"]),
            "HISTORY": jdump(context["history"]),
            "ALLOWED_DIAGNOSES": jdump(list(ALLOWED_DIAGNOSES)),
            "REPAIR_FEEDBACK": repair_feedback or "(none)",
        })

    def critique(self, context: dict) -> AgentResult:
        feedback = None
        prompt = raw = ""
        last_error = "no attempt made"
        for attempt in range(1, self.max_repair_attempts + 2):
            prompt = self.build_prompt(context, feedback)
            try:
                raw = self.backend.complete(prompt, self.kind)
                clean = validate_critique(extract_json(raw))
            except (ProposalError, BackendError) as exc:
                last_error = str(exc)
                feedback = (f"Your previous reply was rejected. Error: {last_error}\n"
                            "Return a corrected JSON object only.")
                continue
            return AgentResult(True, clean, prompt, raw, attempt)
        return AgentResult(False, _fallback_critique(last_error), prompt, raw,
                           self.max_repair_attempts + 1, last_error)


def validate_critique(obj: Any) -> dict:
    if not isinstance(obj, dict):
        raise ProposalError("Critique must be a JSON object.")
    out = {}
    for key in ("diagnosis", "evidence", "recommendations"):
        val = obj.get(key)
        if isinstance(val, str):
            val = [val]
        if not isinstance(val, list) or not val:
            raise ProposalError(
                f"Critique key {key!r} must be a non-empty list of strings. Required keys: "
                "diagnosis, evidence, recommendations."
            )
        out[key] = [str(v)[:400] for v in val[:MAX_ITEMS]]
    bad = [d for d in out["diagnosis"] if d not in ALLOWED_DIAGNOSES]
    if bad:
        raise ProposalError(
            f"Diagnosis label(s) {bad} are not allowed. Use only: {list(ALLOWED_DIAGNOSES)}"
        )
    return out


def _fallback_critique(error: str) -> dict:
    """Keeps the loop alive when the critic backend fails; recorded as such."""
    return {
        "diagnosis": ["ok"],
        "evidence": [f"critic unavailable: {error[:200]}"],
        "recommendations": ["explore a different model family or feature set"],
        "_fallback": True,
    }
