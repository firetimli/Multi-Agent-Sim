"""Shared helpers: config loading, seeding, hashing, templating, JSON extraction.

Trusted infrastructure. Agents never execute code from this module directly;
it is used by the orchestrator and runner only.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


class ProposalError(ValueError):
    """Raised when an agent proposal violates the trusted contract.

    The message is fed back to the agent as repair feedback, so it must be
    specific and actionable (say what was wrong AND what is allowed).
    """


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def load_config(path: str | Path) -> dict:
    with open(path, "r") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_path"] = str(path)
    return cfg


def resolve_path(p: str | Path) -> Path:
    """Resolve a config path relative to the repo root unless absolute."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------- #
# determinism
# --------------------------------------------------------------------------- #
def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))


# --------------------------------------------------------------------------- #
# hashing
# --------------------------------------------------------------------------- #
def sha1_obj(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha1(payload).hexdigest()[:16]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# io / templating
# --------------------------------------------------------------------------- #
def read_text(path: str | Path) -> str:
    with open(path, "r") as fh:
        return fh.read()


def write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w") as fh:
        fh.write(text)


def write_json(path: str | Path, obj: Any) -> None:
    write_text(path, jdump(obj))


def jdump(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=False, default=_json_default)


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


def render(template: str, mapping: dict[str, str]) -> str:
    """Substitute {{KEY}} placeholders. Chosen over str.format/jinja because
    the prompts contain JSON braces."""
    out = template
    for key, value in mapping.items():
        out = out.replace("{{" + key + "}}", value if isinstance(value, str) else jdump(value))
    return out


# --------------------------------------------------------------------------- #
# JSON extraction from LLM output
# --------------------------------------------------------------------------- #
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model response.

    Tries, in order: whole string, fenced blocks, first balanced {...} run.
    Raises ProposalError with the raw tail on failure so the agent can repair.
    """
    candidates: list[str] = []
    stripped = text.strip()
    if stripped.startswith("{"):
        candidates.append(stripped)
    candidates.extend(m.group(1).strip() for m in _FENCE_RE.finditer(text))
    balanced = _first_balanced_object(text)
    if balanced:
        candidates.append(balanced)

    for cand in candidates:
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ProposalError(
        "Response did not contain a parseable JSON object. "
        "Reply with a single JSON object and nothing else. "
        f"Received (last 400 chars): {text[-400:]!r}"
    )


def _first_balanced_object(text: str) -> str | None:
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        start = text.find("{", start + 1)
    return None
