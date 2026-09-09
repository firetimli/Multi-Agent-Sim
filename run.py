#!/usr/bin/env python3
"""Entrypoint for Version 0.

    # 3-experiment smoke test, no data download, no API calls
    python run.py --config configs/synthetic.yaml

    # inspect the dataset profile the Modeling Agent will see
    python run.py --config configs/retailrocket.yaml --profile-only

    # single manual proposal through the trusted path (implementation step 2)
    python run.py --config configs/retailrocket.yaml --proposal configs/example_proposal.json

    # the real loop
    python run.py --config configs/retailrocket.yaml --backend codex_cli

    # leaderboard over everything logged so far
    python run.py --leaderboard
"""
from __future__ import annotations

import argparse
import json
import sys

from src.dataset import load_dataset
from src.orchestrator import Orchestrator, run_single_proposal
from src.store import ExperimentStore, format_leaderboard, leaderboard
from src.utils import jdump, load_config, resolve_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Multi-agent AutoML for user behaviour prediction (v0)")
    ap.add_argument("--config", default="configs/synthetic.yaml")
    ap.add_argument("--backend", choices=["codex_cli", "mock"], default=None,
                    help="override agent.backend from the config")
    ap.add_argument("--max-experiments", type=int, default=None,
                    help="override budget.max_experiments")
    ap.add_argument("--profile-only", action="store_true",
                    help="build the dataset and print the trusted profile, then exit")
    ap.add_argument("--proposal", default=None,
                    help="run one proposal JSON file through the trusted path and exit")
    ap.add_argument("--leaderboard", action="store_true",
                    help="print the leaderboard from runs/experiments.jsonl and exit")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.leaderboard:
        cfg = load_config(resolve_path(args.config))
        store = ExperimentStore(resolve_path(cfg["paths"]["experiments_file"]))
        print(format_leaderboard(leaderboard(list(store.iter_all())), limit=100))
        return 0

    cfg = load_config(resolve_path(args.config))

    if args.profile_only:
        ds = load_dataset(cfg, verbose=not args.quiet)
        print(jdump({"profile": ds.profile, "split_manifest": ds.split_manifest}))
        return 0

    if args.proposal:
        with open(resolve_path(args.proposal)) as fh:
            proposal = json.load(fh)
        result = run_single_proposal(cfg, proposal, verbose=not args.quiet)
        return 0 if result["status"] == "success" else 1

    orch = Orchestrator(cfg, backend_override=args.backend,
                        max_experiments=args.max_experiments, verbose=not args.quiet)
    summary = orch.run()
    return 0 if summary["n_success"] > 0 else 1


def _cli() -> int:
    try:
        return main()
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(_cli())
