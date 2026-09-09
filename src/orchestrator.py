"""Deterministic experiment controller.

TRUSTED INFRASTRUCTURE. There is no LLM manager in Version 0: budget
enforcement, parent selection, stopping criteria and best-model selection are
plain code, because a language model that can decide "one more experiment" or
"this run looks best" is a reproducibility and integrity hole.

Loop:
    context (profile + history + critique)
      -> ModelingAgent proposal (validated before budget is spent)
      -> ExperimentRunner: features -> train -> validation metrics
      -> record appended to runs/experiments.jsonl
      -> CriticAgent critique
      -> next iteration
"""
from __future__ import annotations

import datetime as _dt
import time
from pathlib import Path
from typing import Any

from .agents import CriticAgent, ModelingAgent, make_backend
from .dataset import Dataset, load_dataset
from .evaluator import PRIMARY_METRIC
from .runner import ExperimentRunner, validate_proposal
from .store import Budget, BudgetExhausted, ExperimentStore, best_of, format_leaderboard, leaderboard
from .utils import ensure_dir, resolve_path, sha256_text, set_seeds, write_json, write_text

HISTORY_WINDOW = 5


class Orchestrator:
    def __init__(self, cfg: dict, backend_override: str | None = None,
                 max_experiments: int | None = None, verbose: bool = True):
        self.cfg = cfg
        self.verbose = verbose
        self.seed = int(cfg.get("run", {}).get("seed", 42))
        set_seeds(self.seed)

        self.ds: Dataset = load_dataset(cfg, verbose=verbose)
        self.run_id = self._make_run_id(backend_override)

        paths = cfg["paths"]
        self.runs_dir = ensure_dir(resolve_path(paths["runs_dir"]))
        self.generated_dir = ensure_dir(resolve_path(paths["generated_dir"]) / self.ds.name / self.run_id)
        self.prompt_dir = ensure_dir(self.runs_dir / "prompts" / self.run_id)
        self.response_dir = ensure_dir(self.runs_dir / "responses" / self.run_id)

        self.store = ExperimentStore(resolve_path(paths["experiments_file"]), run_id=self.run_id)
        self.budget = Budget(int(max_experiments or cfg["budget"]["max_experiments"]))
        self.patience = cfg["budget"].get("patience")

        self.runner = ExperimentRunner(
            self.ds, self.generated_dir, resolve_path(paths["interim_dir"]))

        agent_cfg = dict(cfg.get("agent", {}))
        self.backend = make_backend(agent_cfg, backend_override)
        self.modeling_agent = ModelingAgent(
            self.backend, resolve_path(agent_cfg.get("modeling_prompt", "prompts/modeling.md")),
            max_repair_attempts=int(agent_cfg.get("max_repair_attempts", 2)))
        self.critic_agent = CriticAgent(
            self.backend, resolve_path(agent_cfg.get("critic_prompt", "prompts/critic.md")),
            max_repair_attempts=int(agent_cfg.get("max_repair_attempts", 2)))
        self.critic_enabled = bool(agent_cfg.get("critic_enabled", True))

        self.latest_critique: dict | None = None
        self.last_issues: dict = {"warnings": [], "error": None}
        self._write_run_config(backend_override)

    # ------------------------------------------------------------------ #
    def _make_run_id(self, backend_override: str | None) -> str:
        stamp = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        backend = backend_override or self.cfg.get("agent", {}).get("backend", "mock")
        return f"{self.ds.name}__{backend}__s{self.seed}__{stamp}"

    def _write_run_config(self, backend_override: str | None) -> None:
        write_json(self.runs_dir / "run_configs" / f"{self.run_id}.json", {
            "run_id": self.run_id,
            "config": {k: v for k, v in self.cfg.items() if not k.startswith("_")},
            "config_path": self.cfg.get("_config_path"),
            "backend": self.backend.name,
            "backend_override": backend_override,
            "seed": self.seed,
            "budget": self.budget.as_dict(),
            "split_manifest": self.ds.split_manifest,
            "dataset_profile": self.ds.profile,
            "prompt_hashes": {
                "modeling": sha256_text(self.modeling_agent.template),
                "critic": sha256_text(self.critic_agent.template),
            },
            "version": "v0",
        })

    # ------------------------------------------------------------------ #
    # the loop
    # ------------------------------------------------------------------ #
    def run(self) -> dict:
        t_start = time.time()
        no_improve = 0
        best_score = -1.0

        while True:
            try:
                self.budget.check()
            except BudgetExhausted:
                self._log("budget exhausted, stopping")
                break

            exp_id = f"exp_{self.budget.used + 1:03d}"
            self._log(f"--- {exp_id} (budget {self.budget.used}/{self.budget.max_experiments}) ---")

            record = self._one_experiment(exp_id)
            self.budget.consume()
            self.store.append(record)

            score = (record.get("val_metrics") or {}).get(PRIMARY_METRIC)
            if score is not None and score > best_score:
                best_score, no_improve = score, 0
            else:
                no_improve += 1
            if self.patience and no_improve >= int(self.patience):
                self._log(f"no improvement in {no_improve} experiments, stopping early")
                break

        return self._finalise(t_start)

    def _one_experiment(self, exp_id: str) -> dict:
        context = self._modeling_context()
        agent_out = self.modeling_agent.propose(context, validate_proposal)
        self._save_io(exp_id, "modeling", agent_out)

        record: dict[str, Any] = {
            "run_id": self.run_id,
            "dataset": self.ds.name,
            "experiment_id": exp_id,
            "parent_experiment": None,
            "proposal": None,
            "train_metrics": {},
            "val_metrics": {},
            "critique": None,
            "seed": self.seed,
            "runtime_seconds": 0.0,
            "status": "invalid_proposal",
            "created_at": _dt.datetime.utcnow().isoformat() + "Z",
            "backend": self.backend.name,
            "split_manifest_hash": self.ds.split_manifest["hash"],
            "agent_attempts": {"modeling": agent_out.attempts},
            "artifacts": {
                "prompt": str(self.prompt_dir / f"{exp_id}.modeling.txt"),
                "response": str(self.response_dir / f"{exp_id}.modeling.txt"),
            },
        }

        if not agent_out.ok:
            record["error"] = agent_out.error
            self.last_issues = {"warnings": [], "error": agent_out.error}
            self._log(f"  proposal rejected after {agent_out.attempts} attempts: {agent_out.error}")
            record["critique"] = self._critique(record, {}) if self.critic_enabled else None
            return record

        proposal = agent_out.payload
        record["parent_experiment"] = proposal.get("parent_experiment")
        self._log(f"  {proposal['model']['family']} / {len(proposal['features'])} features"
                  f" :: {proposal['rationale'][:90]}")

        result = self.runner.run(proposal, exp_id, self.seed)
        record.update({
            "proposal": {k: v for k, v in proposal.items() if not k.startswith("_")},
            "proposal_hash": proposal["proposal_hash"],
            "status": result["status"],
            "train_metrics": result.get("train_metrics", {}),
            "val_metrics": result.get("val_metrics", {}),
            "overfit_gap": result.get("overfit_gap"),
            "n_features": result.get("n_features"),
            "feature_names": result.get("feature_names"),
            "model_info": result.get("model_info"),
            "runtime_seconds": result["runtime_seconds"],
            "feature_seconds": result.get("feature_seconds"),
            "train_seconds": result.get("train_seconds"),
            "warnings": result.get("warnings", []),
            "error": result.get("error"),
        })
        record["artifacts"]["generated_dir"] = result["generated_dir"]
        self.last_issues = {"warnings": record["warnings"], "error": record["error"]}

        vm = record["val_metrics"] or {}
        tm = record["train_metrics"] or {}
        self._log(f"  status={record['status']} train_auroc={_f(tm.get('auroc'))} "
                  f"val_auroc={_f(vm.get('auroc'))} val_auprc={_f(vm.get('auprc'))} "
                  f"({record['runtime_seconds']}s)")

        record["critique"] = self._critique(record, result) if self.critic_enabled else None
        return record

    # ------------------------------------------------------------------ #
    # context assembly (the only channel through which agents see the past)
    # ------------------------------------------------------------------ #
    def _modeling_context(self) -> dict:
        return {
            "dataset_profile": self.ds.profile,
            "budget": self.budget.as_dict(),
            "best_so_far": self._best_summary(),
            "history": self._history(),
            "latest_critique": self.latest_critique or "(no experiments yet)",
            "last_issues": self.last_issues,
        }

    def _critique(self, record: dict, result: dict) -> dict:
        ctx = {
            "dataset_profile": self.ds.profile,
            "proposal": record.get("proposal") or "(no valid proposal was produced)",
            "train_metrics": record.get("train_metrics") or {},
            "val_metrics": record.get("val_metrics") or {},
            "overfit_gap": record.get("overfit_gap"),
            "run_info": {
                "status": record["status"],
                "runtime_seconds": record.get("runtime_seconds"),
                "warnings": record.get("warnings", []),
                "error": record.get("error"),
                "model_info": record.get("model_info"),
            },
            "best_so_far": self._best_summary(),
            "history": self._history(),
        }
        out = self.critic_agent.critique(ctx)
        self._save_io(record["experiment_id"], "critic", out)
        record.setdefault("agent_attempts", {})["critic"] = out.attempts
        record["artifacts"]["critic_prompt"] = str(
            self.prompt_dir / f"{record['experiment_id']}.critic.txt")
        self.latest_critique = dict(out.payload, experiment_id=record["experiment_id"])
        diag = ", ".join(out.payload.get("diagnosis", []))
        self._log(f"  critique: {diag}")
        return out.payload

    def _history(self) -> list[dict]:
        items = []
        for r in self.store.records_this_run()[-HISTORY_WINDOW:]:
            prop = r.get("proposal") or {}
            items.append({
                "experiment_id": r["experiment_id"],
                "parent_experiment": r.get("parent_experiment"),
                "status": r["status"],
                "features": [f["name"] + (f"({f['parameters']})" if f["parameters"] else "")
                             for f in prop.get("features", [])],
                "model": prop.get("model"),
                "train_auroc": (r.get("train_metrics") or {}).get("auroc"),
                "val_auroc": (r.get("val_metrics") or {}).get("auroc"),
                "val_auprc": (r.get("val_metrics") or {}).get("auprc"),
                "val_f1": (r.get("val_metrics") or {}).get("f1"),
                "runtime_seconds": r.get("runtime_seconds"),
                "error": r.get("error"),
                "critique": (r.get("critique") or {}).get("diagnosis"),
            })
        return items

    def _best_summary(self) -> Any:
        best = self.store.best_this_run(PRIMARY_METRIC)
        if not best:
            return "(nothing succeeded yet)"
        return {
            "experiment_id": best["experiment_id"],
            "val_auroc": best["val_metrics"]["auroc"],
            "val_auprc": best["val_metrics"].get("auprc"),
            "model": (best.get("proposal") or {}).get("model"),
            "features": [f["name"] for f in (best.get("proposal") or {}).get("features", [])],
        }

    # ------------------------------------------------------------------ #
    def _save_io(self, exp_id: str, agent: str, out) -> None:
        write_text(self.prompt_dir / f"{exp_id}.{agent}.txt", out.prompt)
        write_text(self.response_dir / f"{exp_id}.{agent}.txt", out.raw_response)

    def _finalise(self, t_start: float) -> dict:
        records = self.store.records_this_run()
        best = best_of(records, PRIMARY_METRIC)
        rows = leaderboard(records, PRIMARY_METRIC)
        summary = {
            "run_id": self.run_id,
            "dataset": self.ds.name,
            "backend": self.backend.name,
            "budget": self.budget.as_dict(),
            "n_success": sum(1 for r in records if r["status"] == "success"),
            "n_failed": sum(1 for r in records if r["status"] != "success"),
            "best_experiment": best["experiment_id"] if best else None,
            "best_val_auroc": best["val_metrics"]["auroc"] if best else None,
            "best_proposal": (best.get("proposal") if best else None),
            "wall_clock_seconds": round(time.time() - t_start, 1),
            "backend_calls": self.backend.calls,
            "selection_rule": "max validation AUROC; test set never evaluated in v0",
        }
        write_json(self.runs_dir / "summaries" / f"{self.run_id}.json", summary)
        write_text(self.runs_dir / "summaries" / f"{self.run_id}.leaderboard.txt",
                   format_leaderboard(rows))
        if self.verbose:
            print("\n" + format_leaderboard(rows))
            print(f"\nbest: {summary['best_experiment']} "
                  f"val_auroc={_f(summary['best_val_auroc'])} "
                  f"({summary['n_success']}/{self.budget.used} succeeded, "
                  f"{summary['wall_clock_seconds']}s)")
        return summary

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)


def _f(v) -> str:
    return "-" if v is None else f"{v:.4f}"


# --------------------------------------------------------------------------- #
# single manual proposal (implementation step 2 / debugging)
# --------------------------------------------------------------------------- #
def run_single_proposal(cfg: dict, proposal: dict, verbose: bool = True) -> dict:
    seed = int(cfg.get("run", {}).get("seed", 42))
    set_seeds(seed)
    ds = load_dataset(cfg, verbose=verbose)
    clean = validate_proposal(proposal)
    gen = ensure_dir(resolve_path(cfg["paths"]["generated_dir"]) / ds.name / "manual")
    runner = ExperimentRunner(ds, gen, resolve_path(cfg["paths"]["interim_dir"]))
    result = runner.run(clean, "manual_001", seed)
    if verbose:
        print(f"status={result['status']}")
        if result.get("error"):
            print(f"error={result['error']}")
        for split in ("train_metrics", "val_metrics"):
            print(f"{split}: {result.get(split)}")
        if result.get("warnings"):
            print(f"warnings: {result['warnings']}")
    return result
