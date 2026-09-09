"""Dataset loading, label construction, and the fixed temporal split.

TRUSTED INFRASTRUCTURE -- agents may not influence anything in this file.

Everything about the prediction task lives here so it can be changed in ONE
place later (label definition is deliberately config-driven, not hard-coded):

  * which column is user / item / timestamp / event type
  * which events count as positive
  * the history window used to build features
  * the prediction horizon

Task formulation (V0): user-level binary behaviour prediction.
  For a cutoff time c, a row exists for every user with >= min_history_events
  in [c - history_days, c).  label = 1 iff that user emits a positive event in
  (c, c + horizon_days].

Split: three consecutive, non-overlapping label windows at the END of the
timeline (train, val, test).  Test rows are materialised so the split never has
to be recomputed later, but V0 never reads their labels -- see split_manifest.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import ensure_dir, resolve_path, sha1_obj

DAY = 86_400


# --------------------------------------------------------------------------- #
# task spec
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TaskSpec:
    user_col: str
    time_col: str
    item_col: str | None
    event_col: str | None
    positive_events: tuple[str, ...]
    event_vocab: tuple[str, ...]
    time_unit: str = "ms"            # "ms" | "s"
    history_days: float = 14.0
    horizon_days: float = 7.0
    min_history_events: int = 1
    max_users_per_split: int | None = 50_000
    seed: int = 42

    @property
    def history_seconds(self) -> int:
        return int(self.history_days * DAY)

    @property
    def horizon_seconds(self) -> int:
        return int(self.horizon_days * DAY)

    def fingerprint(self) -> str:
        return sha1_obj(asdict(self))


def task_from_config(cfg: dict) -> TaskSpec:
    cols = cfg["data"]["columns"]
    task = cfg["task"]
    return TaskSpec(
        user_col=cols["user"],
        time_col=cols["time"],
        item_col=cols.get("item"),
        event_col=cols.get("event_type"),
        positive_events=tuple(task["positive_events"]),
        event_vocab=tuple(task["event_vocab"]),
        time_unit=cfg["data"].get("time_unit", "ms"),
        history_days=float(task["history_days"]),
        horizon_days=float(task["horizon_days"]),
        min_history_events=int(task.get("min_history_events", 1)),
        max_users_per_split=task.get("max_users_per_split"),
        seed=int(cfg.get("run", {}).get("seed", 42)),
    )


# --------------------------------------------------------------------------- #
# cutoff-safe history access
# --------------------------------------------------------------------------- #
class UserEventIndex:
    """Per-user, time-sorted event arrays with hard cutoff enforcement.

    This is the ONLY way feature code reaches the event log.  `history()`
    asserts that no returned event is at or after the cutoff, which makes
    "features cannot see the future" a property of the code rather than of
    reviewer diligence.
    """

    def __init__(self, users: np.ndarray, ts: np.ndarray, items: np.ndarray, etypes: np.ndarray):
        order = np.lexsort((ts, users))
        self.users = users[order]
        self.ts = ts[order]
        self.items = items[order]
        self.etypes = etypes[order]
        # contiguous block boundaries per user code
        self.n_users = int(self.users.max()) + 1 if len(self.users) else 0
        self.starts = np.searchsorted(self.users, np.arange(self.n_users), side="left")
        self.ends = np.searchsorted(self.users, np.arange(self.n_users), side="right")

    def history(self, user: int, cutoff_ts: int, window_start_ts: int):
        """Events for `user` with window_start_ts <= ts < cutoff_ts."""
        lo, hi = self.starts[user], self.ends[user]
        block = self.ts[lo:hi]
        a = lo + int(np.searchsorted(block, window_start_ts, side="left"))
        b = lo + int(np.searchsorted(block, cutoff_ts, side="left"))
        ts = self.ts[a:b]
        if len(ts) and ts[-1] >= cutoff_ts:  # pragma: no cover - invariant guard
            raise AssertionError("history() leaked an event at/after the cutoff")
        return ts, self.items[a:b], self.etypes[a:b]


@dataclass
class Dataset:
    name: str
    task: TaskSpec
    index: UserEventIndex
    rows: pd.DataFrame                    # user, cutoff_ts, label, split
    profile: dict[str, Any]
    split_manifest: dict[str, Any]
    n_event_types: int
    cache_key: str = ""
    user_ids: np.ndarray = field(default_factory=lambda: np.array([]))

    def split(self, name: str) -> pd.DataFrame:
        if name == "test":
            raise PermissionError(
                "Version 0 does not evaluate on test. Test rows exist only so the "
                "split never has to be recomputed."
            )
        return self.rows[self.rows["split"] == name].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_dataset(cfg: dict, verbose: bool = True) -> Dataset:
    task = task_from_config(cfg)
    name = cfg["name"]
    source = cfg["data"].get("source", "csv")
    cache_dir = ensure_dir(resolve_path(cfg["paths"]["interim_dir"]))
    key = sha1_obj({"name": name, "source": source, "task": asdict(task),
                    "path": cfg["data"].get("events_path"),
                    "synthetic": cfg["data"].get("synthetic", {})})
    cache_file = cache_dir / f"dataset_{name}_{key}.pkl"

    if cache_file.exists() and cfg["data"].get("use_cache", True):
        if verbose:
            print(f"[dataset] cache hit {cache_file.name}")
        bundle = pd.read_pickle(cache_file)
    else:
        t0 = time.time()
        events = _load_events(cfg, task, verbose=verbose)
        bundle = _build_bundle(name, events, task, verbose=verbose)
        pd.to_pickle(bundle, cache_file)
        if verbose:
            print(f"[dataset] built in {time.time() - t0:.1f}s -> {cache_file.name}")

    index = UserEventIndex(bundle["users"], bundle["ts"], bundle["items"], bundle["etypes"])
    return Dataset(
        name=name,
        task=task,
        index=index,
        rows=bundle["rows"],
        profile=bundle["profile"],
        split_manifest=bundle["split_manifest"],
        n_event_types=len(task.event_vocab),
        cache_key=key,
        user_ids=bundle["user_ids"],
    )


def _load_events(cfg: dict, task: TaskSpec, verbose: bool) -> pd.DataFrame:
    source = cfg["data"].get("source", "csv")
    if source == "synthetic":
        return _make_synthetic_events(cfg, task, verbose=verbose)
    if source != "csv":
        raise ValueError(f"unknown data.source {source!r}")

    path = resolve_path(cfg["data"]["events_path"])
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download RetailRocket events.csv (see README) or run "
            f"with configs/synthetic.yaml."
        )
    usecols = [task.user_col, task.time_col]
    if task.item_col:
        usecols.append(task.item_col)
    if task.event_col:
        usecols.append(task.event_col)
    if verbose:
        print(f"[dataset] reading {path}")
    df = pd.read_csv(path, usecols=usecols)
    return df


def _build_bundle(name: str, df: pd.DataFrame, task: TaskSpec, verbose: bool) -> dict:
    # --- normalise ---------------------------------------------------------
    ts = df[task.time_col].to_numpy(dtype="int64")
    if task.time_unit == "ms":
        ts = ts // 1000
    elif task.time_unit != "s":
        raise ValueError(f"unsupported time_unit {task.time_unit!r}")

    user_codes, user_ids = pd.factorize(df[task.user_col], sort=False)
    user_codes = user_codes.astype("int64")

    if task.item_col:
        item_codes = pd.factorize(df[task.item_col], sort=False)[0].astype("int64")
    else:
        item_codes = np.zeros(len(df), dtype="int64")

    vocab = {name_: i for i, name_ in enumerate(task.event_vocab)}
    if task.event_col:
        etypes = df[task.event_col].map(vocab).fillna(-1).to_numpy(dtype="int8")
    else:
        etypes = np.zeros(len(df), dtype="int8")

    pos_ids = np.array([vocab[e] for e in task.positive_events if e in vocab], dtype="int8")
    if len(pos_ids) == 0:
        raise ValueError(
            f"positive_events {task.positive_events} are not in event_vocab {task.event_vocab}"
        )

    # --- labels + split ----------------------------------------------------
    rows, manifest = _build_rows(user_codes, ts, etypes, pos_ids, task, verbose=verbose)
    profile = _profile(user_codes, ts, etypes, task, rows, manifest)
    return {
        "users": user_codes, "ts": ts, "items": item_codes, "etypes": etypes,
        "user_ids": np.asarray(user_ids), "rows": rows,
        "profile": profile, "split_manifest": manifest,
    }


def _build_rows(users, ts, etypes, pos_ids, task: TaskSpec, verbose: bool):
    n_users = int(users.max()) + 1
    t_max = int(ts.max())
    hz = task.horizon_seconds
    cutoffs = {"train": t_max - 3 * hz, "val": t_max - 2 * hz, "test": t_max - hz}
    if cutoffs["train"] - task.history_seconds < int(ts.min()):
        raise ValueError(
            "Timeline too short for history_days + 3 * horizon_days. "
            f"span={(t_max - int(ts.min())) / DAY:.1f}d, "
            f"needed={(task.history_seconds + 3 * hz) / DAY:.1f}d"
        )

    is_pos = np.isin(etypes, pos_ids)
    rng = np.random.default_rng(task.seed)
    frames, manifest_splits = [], {}

    for split, cutoff in cutoffs.items():
        hist_start = cutoff - task.history_seconds
        hist_mask = (ts >= hist_start) & (ts < cutoff)
        hist_counts = np.bincount(users[hist_mask], minlength=n_users)
        eligible = np.flatnonzero(hist_counts >= task.min_history_events)

        fut_mask = (ts > cutoff) & (ts <= cutoff + hz) & is_pos
        pos_counts = np.bincount(users[fut_mask], minlength=n_users)
        labels = (pos_counts[eligible] > 0).astype("int8")

        if task.max_users_per_split and len(eligible) > task.max_users_per_split:
            # uniform subsample: preserves the positive rate of the full cohort
            pick = rng.choice(len(eligible), size=task.max_users_per_split, replace=False)
            pick.sort()
            eligible, labels = eligible[pick], labels[pick]

        frames.append(pd.DataFrame({
            "user": eligible.astype("int64"),
            "cutoff_ts": np.full(len(eligible), cutoff, dtype="int64"),
            "label": labels,
            "split": split,
        }))
        manifest_splits[split] = {
            "cutoff_ts": int(cutoff),
            "cutoff_utc": pd.to_datetime(cutoff, unit="s").isoformat(),
            "history_window": [int(hist_start), int(cutoff)],
            "label_window": [int(cutoff), int(cutoff + hz)],
            "n_rows": int(len(eligible)),
            "n_positive": int(labels.sum()),
        }
        if verbose:
            rate = labels.mean() if len(labels) else float("nan")
            print(f"[dataset] {split:5s} rows={len(eligible):7d} pos_rate={rate:.4f}")

    rows = pd.concat(frames, ignore_index=True)
    manifest = {
        "task": asdict(task),
        "splits": manifest_splits,
        "note": "test labels are not read anywhere in Version 0",
    }
    manifest["hash"] = sha1_obj(manifest)
    return rows, manifest


def _profile(users, ts, etypes, task: TaskSpec, rows: pd.DataFrame, manifest: dict) -> dict:
    """Trusted dataset summary shown to the Modeling Agent.

    Deliberately excludes every test-split statistic.
    """
    vocab = list(task.event_vocab)
    counts = np.bincount(etypes[etypes >= 0], minlength=len(vocab))
    per_user = np.bincount(users)
    per_user = per_user[per_user > 0]
    dev = rows[rows["split"].isin(["train", "val"])]
    profile = {
        "n_events": int(len(ts)),
        "n_users": int(len(per_user)),
        "n_event_types": len(vocab),
        "event_type_counts": {v: int(c) for v, c in zip(vocab, counts)},
        "events_per_user": {
            "mean": round(float(per_user.mean()), 2),
            "p50": int(np.percentile(per_user, 50)),
            "p90": int(np.percentile(per_user, 90)),
            "p99": int(np.percentile(per_user, 99)),
            "max": int(per_user.max()),
        },
        "time_range_utc": [
            pd.to_datetime(int(ts.min()), unit="s").isoformat(),
            pd.to_datetime(int(ts.max()), unit="s").isoformat(),
        ],
        "task": {
            "type": "binary user behaviour prediction",
            "positive_events": list(task.positive_events),
            "history_days": task.history_days,
            "horizon_days": task.horizon_days,
            "min_history_events": task.min_history_events,
        },
        "splits": {
            s: {
                "n_rows": manifest["splits"][s]["n_rows"],
                "positive_rate": round(
                    manifest["splits"][s]["n_positive"] / max(manifest["splits"][s]["n_rows"], 1), 5
                ),
                "cutoff_utc": manifest["splits"][s]["cutoff_utc"],
            }
            for s in ("train", "val")
        },
        "train_val_user_overlap": round(
            float(len(set(dev[dev.split == "train"].user) & set(dev[dev.split == "val"].user))
                  / max(len(set(dev[dev.split == "val"].user)), 1)), 3
        ),
    }
    return profile


# --------------------------------------------------------------------------- #
# synthetic RetailRocket-shaped data (for fast smoke tests, no download)
# --------------------------------------------------------------------------- #
def _make_synthetic_events(cfg: dict, task: TaskSpec, verbose: bool) -> pd.DataFrame:
    p = cfg["data"].get("synthetic", {})
    n_users = int(p.get("n_users", 4000))
    days = float(p.get("days", 60))
    seed = int(p.get("seed", 0))
    n_items = int(p.get("n_items", 2000))
    rng = np.random.default_rng(seed)

    t0 = int(pd.Timestamp("2015-06-01").timestamp())
    span = int(days * DAY)
    # latent activity level drives both event volume and purchase propensity,
    # so history-derived features are genuinely predictive.
    rate = rng.lognormal(mean=1.6, sigma=0.9, size=n_users)
    buy_p = np.clip(0.004 * rate + rng.normal(0, 0.004, n_users), 0.0005, 0.25)
    focus = rng.beta(2, 2, size=n_users)  # item concentration

    users, tss, items, evs = [], [], [], []
    for u in range(n_users):
        n = rng.poisson(rate[u] * days / 7.0 * 3.0) + 1
        t = np.sort(rng.integers(t0, t0 + span, size=n))
        favourite = rng.integers(0, n_items)
        use_fav = rng.random(n) < focus[u]
        it = np.where(use_fav, favourite, rng.integers(0, n_items, size=n))
        r = rng.random(n)
        ev = np.where(r < buy_p[u], 2, np.where(r < buy_p[u] + 0.06, 1, 0))
        users.append(np.full(n, u))
        tss.append(t)
        items.append(it)
        evs.append(ev)

    vocab = list(task.event_vocab)
    df = pd.DataFrame({
        task.user_col: np.concatenate(users),
        task.time_col: np.concatenate(tss),
        task.item_col or "item": np.concatenate(items),
        task.event_col or "event": [vocab[i] for i in np.concatenate(evs)],
    })
    if verbose:
        print(f"[dataset] synthetic events={len(df)} users={n_users}")
    return df
