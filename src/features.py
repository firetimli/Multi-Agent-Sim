"""Trusted feature catalog and feature-matrix builder.

TRUSTED INFRASTRUCTURE -- agents SELECT and CONFIGURE features from
FEATURE_CATALOG; they cannot add or edit feature code in Version 0.

Every feature is computed from `UserEventIndex.history(user, cutoff, window)`,
which cannot return events at or after the row's cutoff.  Leakage-safety is
therefore structural, not a review convention.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .dataset import Dataset
from .utils import ProposalError, ensure_dir, sha1_obj

MAX_FEATURES = 12  # per proposal, before expansion into columns


@dataclass(frozen=True)
class Param:
    kind: str                     # "float" | "int" | "bool"
    default: Any
    low: float | None = None
    high: float | None = None
    doc: str = ""


@dataclass(frozen=True)
class Feature:
    doc: str
    params: dict[str, Param]
    n_columns: str = "1"          # human-readable, for the prompt


FEATURE_CATALOG: dict[str, Feature] = {
    "history_count": Feature(
        "Number of events by the user inside the history window.", {}),
    "unique_item_count": Feature(
        "Number of distinct items the user interacted with in the history window.", {}),
    "time_since_last_event": Feature(
        "Seconds between the user's last event and the cutoff (recency).", {}),
    "mean_inter_event_time": Feature(
        "Mean seconds between consecutive events; NaN if fewer than 2 events.", {}),
    "recent_event_count": Feature(
        "Event count in the last `window_days` before the cutoff (short-term intensity).",
        {"window_days": Param("float", 1.0, 0.05, 30.0, "length of the recent window in days")}),
    "item_repeat_ratio": Feature(
        "1 - unique_items / events; how repetitive the user's browsing is.", {}),
    "session_length": Feature(
        "Mean number of events per session, sessionised by a `gap_minutes` inactivity gap.",
        {"gap_minutes": Param("float", 30.0, 1.0, 720.0, "inactivity gap that ends a session")}),
    "event_type_counts": Feature(
        "One column per event type (e.g. view / addtocart / transaction). Set "
        "`normalize` to emit shares instead of raw counts.",
        {"normalize": Param("bool", False, doc="divide counts by total events")},
        n_columns="one per event type"),
}


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def validate_feature_spec(features: Any) -> list[dict]:
    """Validate + normalise the agent's `features` list. Raises ProposalError."""
    if not isinstance(features, list) or not features:
        raise ProposalError(
            "`features` must be a non-empty list of {\"name\": ..., \"parameters\": {...}} "
            f"objects. Allowed names: {sorted(FEATURE_CATALOG)}"
        )
    if len(features) > MAX_FEATURES:
        raise ProposalError(f"At most {MAX_FEATURES} features per proposal, got {len(features)}.")

    clean: list[dict] = []
    seen: set[str] = set()
    for i, item in enumerate(features):
        if isinstance(item, str):
            item = {"name": item, "parameters": {}}
        if not isinstance(item, dict) or "name" not in item:
            raise ProposalError(f"features[{i}] must be an object with a \"name\" key.")
        name = item["name"]
        if name not in FEATURE_CATALOG:
            raise ProposalError(
                f"Unknown feature {name!r}. You may only use features from the catalog: "
                f"{sorted(FEATURE_CATALOG)}"
            )
        spec = FEATURE_CATALOG[name]
        raw = item.get("parameters") or {}
        if not isinstance(raw, dict):
            raise ProposalError(f"features[{i}].parameters must be an object.")
        unknown = set(raw) - set(spec.params)
        if unknown:
            raise ProposalError(
                f"Feature {name!r} got unknown parameter(s) {sorted(unknown)}; "
                f"allowed: {sorted(spec.params) or 'none'}"
            )
        params = {}
        for pname, pspec in spec.params.items():
            params[pname] = _coerce_param(name, pname, pspec, raw.get(pname, pspec.default))
        entry = {"name": name, "parameters": params}
        key = sha1_obj(entry)
        if key in seen:
            continue  # silently drop exact duplicates
        seen.add(key)
        clean.append(entry)
    return clean


def _coerce_param(fname: str, pname: str, spec: Param, value: Any):
    if spec.kind == "bool":
        return bool(value)
    try:
        value = float(value) if spec.kind == "float" else int(value)
    except (TypeError, ValueError):
        raise ProposalError(
            f"Feature {fname!r} parameter {pname!r} must be a {spec.kind}, got {value!r}."
        )
    if spec.low is not None:
        value = max(value, spec.low if spec.kind == "float" else int(spec.low))
    if spec.high is not None:
        value = min(value, spec.high if spec.kind == "float" else int(spec.high))
    return value


def catalog_for_prompt() -> list[dict]:
    out = []
    for name, f in FEATURE_CATALOG.items():
        out.append({
            "name": name,
            "description": f.doc,
            "columns": f.n_columns,
            "parameters": {
                p: {"type": s.kind, "default": s.default,
                    **({"range": [s.low, s.high]} if s.low is not None else {}),
                    "doc": s.doc}
                for p, s in f.params.items()
            },
        })
    return out


# --------------------------------------------------------------------------- #
# feature computation
# --------------------------------------------------------------------------- #
def build_features(
    ds: Dataset,
    rows: pd.DataFrame,
    features: list[dict],
    split: str,
    cache_dir: str | Path | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Compute the feature matrix for `rows` (one row per user/cutoff).

    Cached on (feature spec, split, dataset fingerprint) because agents reuse
    feature sets across experiments.
    """
    names = column_names(features, ds)
    key = sha1_obj({"f": features, "split": split, "ds": ds.cache_key,
                    "task": ds.task.fingerprint(), "n": len(rows)})
    cache_file = None
    if cache_dir:
        cache_file = ensure_dir(Path(cache_dir) / "features") / f"{split}_{key}.npy"
        if cache_file.exists():
            return np.load(cache_file), names

    index = ds.index
    n_types = ds.n_event_types
    hist_window = ds.task.history_seconds
    users = rows["user"].to_numpy()
    cutoffs = rows["cutoff_ts"].to_numpy()

    X = np.full((len(rows), len(names)), np.nan, dtype="float32")
    for r in range(len(rows)):
        cutoff = int(cutoffs[r])
        ts, items, etypes = index.history(users[r], cutoff, cutoff - hist_window)
        col = 0
        for feat in features:
            vals = _compute(feat["name"], feat["parameters"], ts, items, etypes, cutoff, n_types)
            X[r, col : col + len(vals)] = vals
            col += len(vals)

    X[~np.isfinite(X) & ~np.isnan(X)] = np.nan  # +/-inf -> nan
    if cache_file is not None:
        np.save(cache_file, X)
    return X, names


def column_names(features: list[dict], ds: Dataset) -> list[str]:
    names: list[str] = []
    for feat in features:
        base = feat["name"]
        suffix = "".join(f"__{k}={v}" for k, v in sorted(feat["parameters"].items()))
        if base == "event_type_counts":
            names.extend(f"{base}__{et}{suffix}" for et in ds.task.event_vocab)
        else:
            names.append(base + suffix)
    return names


def _compute(name, params, ts, items, etypes, cutoff, n_types) -> np.ndarray:
    n = len(ts)
    if name == "history_count":
        return np.array([n], dtype="float32")
    if name == "unique_item_count":
        return np.array([np.unique(items).size if n else np.nan], dtype="float32")
    if name == "time_since_last_event":
        return np.array([cutoff - ts[-1] if n else np.nan], dtype="float32")
    if name == "mean_inter_event_time":
        return np.array([np.diff(ts).mean() if n >= 2 else np.nan], dtype="float32")
    if name == "recent_event_count":
        start = cutoff - int(params["window_days"] * 86_400)
        first = int(np.searchsorted(ts, start, side="left"))
        return np.array([n - first], dtype="float32")
    if name == "item_repeat_ratio":
        if n == 0:
            return np.array([np.nan], dtype="float32")
        return np.array([1.0 - np.unique(items).size / n], dtype="float32")
    if name == "session_length":
        if n == 0:
            return np.array([np.nan], dtype="float32")
        gap = int(params["gap_minutes"] * 60)
        sessions = 1 + int((np.diff(ts) > gap).sum()) if n >= 2 else 1
        return np.array([n / sessions], dtype="float32")
    if name == "event_type_counts":
        counts = np.bincount(etypes[etypes >= 0], minlength=n_types).astype("float32")
        if params.get("normalize") and n:
            counts = counts / n
        return counts
    raise ProposalError(f"Unknown feature {name!r} reached the computer (catalog bug).")
