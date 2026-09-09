"""Modeling Agent: proposes a JSON pipeline spec (features + model + hyperparameters).

Spec-only in Version 0 -- the agent selects from trusted catalogs and cannot emit
code. Stateless per call: all memory arrives as an explicit context block built
from the experiment store, which keeps every decision auditable and replayable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .. import features as features_mod
from .. import models as models_mod
from ..utils import ProposalError, extract_json, jdump, read_text, render
from .backend import Backend, BackendError


@dataclass
class AgentResult:
    ok: bool
    payload: dict
    prompt: str
    raw_response: str
    attempts: int
    error: str | None = None


class ModelingAgent:
    kind = "modeling"

    def __init__(self, backend: Backend, prompt_path: str | Path, max_repair_attempts: int = 2):
        self.backend = backend
        self.template = read_text(prompt_path)
        self.max_repair_attempts = max_repair_attempts

    def build_prompt(self, context: dict, repair_feedback: str | None = None) -> str:
        return render(self.template, {
            "DATASET_PROFILE": jdump(context["dataset_profile"]),
            "FEATURE_CATALOG": jdump(features_mod.catalog_for_prompt()),
            "MODEL_CATALOG": jdump(models_mod.catalog_for_prompt()),
            "BUDGET": jdump(context["budget"]),
            "BEST_SO_FAR": jdump(context["best_so_far"]),
            "HISTORY": jdump(context["history"]),
            "LATEST_CRITIQUE": jdump(context["latest_critique"]),
            "LAST_ISSUES": jdump(context["last_issues"]),
            "REPAIR_FEEDBACK": repair_feedback or "(none)",
        })

    def propose(self, context: dict, validator: Callable[[dict], dict]) -> AgentResult:
        """Ask for a proposal, validating (and repairing) before budget is spent."""
        feedback: str | None = None
        prompt = ""
        raw = ""
        last_error = "no attempt made"
        for attempt in range(1, self.max_repair_attempts + 2):
            prompt = self.build_prompt(context, feedback)
            try:
                raw = self.backend.complete(prompt, self.kind)
                proposal = extract_json(raw)
                clean = validator(proposal)
            except (ProposalError, BackendError) as exc:
                last_error = str(exc)
                feedback = (
                    "Your previous reply was rejected by the trusted validator. "
                    f"Error: {last_error}\nReturn a corrected JSON object only."
                )
                continue
            return AgentResult(True, clean, prompt, raw, attempt)
        return AgentResult(False, {}, prompt, raw, self.max_repair_attempts + 1, last_error)
