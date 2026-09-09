"""Deterministic experiment execution.

TRUSTED INFRASTRUCTURE -- validates an agent proposal against the trusted
catalogs, then builds features, fits the model, and scores train + validation.

Version 0 runs in-process rather than in a subprocess sandbox: proposals are
spec-only, so there is no agent-authored code to isolate. When free-form
pipeline code is introduced, this is the seam where the sandbox goes.

The test split is never touched here; Dataset.split("test") raises.
"""
from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from . import features as features_mod
from . import models as models_mod
from .dataset import Dataset
from .evaluator import compute_metrics, overfit_gap
from .utils import ProposalError, ensure_dir, sha1_obj, write_json

PROPOSAL_KEYS = {"parent_experiment", "rationale", "features", "model"}


# --------------------------------------------------------------------------- #
# proposal contract
# --------------------------------------------------------------------------- #
def validate_proposal(proposal: Any) -> dict:
    """Normalise + validate a proposal. Raises ProposalError with actionable text.

    Called BEFORE any budget is spent, so an agent gets repair rounds for free;
    a proposal that is still invalid after the repair rounds is logged as a
    failed experiment and does consume budget.
    """
    if not isinstance(proposal, dict):
        raise ProposalError("Proposal must be a JSON object.")
    unknown = set(proposal) - PROPOSAL_KEYS
    if unknown:
        raise ProposalError(
            f"Unknown top-level key(s) {sorted(unknown)}. Allowed keys: {sorted(PROPOSAL_KEYS)}"
        )
    if "features" not in proposal or "model" not in proposal:
        raise ProposalError("Proposal must contain both \"features\" and \"model\".")

    clean_features = features_mod.validate_feature_spec(proposal["features"])
    clean_model, warnings = models_mod.validate_model_spec(proposal["model"])
    clean = {
        "parent_experiment": proposal.get("parent_experiment"),
        "rationale": str(proposal.get("rationale", ""))[:2000],
        "features": clean_features,
        "model": clean_model,
    }
    clean["proposal_hash"] = sha1_obj({"f": clean_features, "m": clean_model})
    clean["_warnings"] = warnings
    return clean


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #
class ExperimentRunner:
    def __init__(self, ds: Dataset, generated_dir: str | Path, interim_dir: str | Path):
        self.ds = ds
        self.generated_dir = ensure_dir(generated_dir)
        self.interim_dir = interim_dir
        self.train_rows = ds.split("train")
        self.val_rows = ds.split("val")
        self.y_train = self.train_rows["label"].to_numpy()
        self.y_val = self.val_rows["label"].to_numpy()

    def run(self, proposal: dict, experiment_id: str, seed: int) -> dict:
        """Execute one experiment. Never raises: failures come back as a result
        dict with status != "success" so the loop can log and continue."""
        out_dir = ensure_dir(Path(self.generated_dir) / experiment_id)
        t_start = time.time()
        result: dict[str, Any] = {
            "status": "success",
            "generated_dir": str(Path(self.generated_dir) / experiment_id),
            "warnings": list(proposal.get("_warnings", [])),
            "error": None,
        }
        write_json(out_dir / "proposal.json", {k: v for k, v in proposal.items()
                                               if not k.startswith("_")})
        try:
            t0 = time.time()
            X_tr, names = features_mod.build_features(
                self.ds, self.train_rows, proposal["features"], "train", self.interim_dir)
            X_val, names_val = features_mod.build_features(
                self.ds, self.val_rows, proposal["features"], "val", self.interim_dir)
            assert names == names_val, "feature column mismatch between splits"
            result["feature_seconds"] = round(time.time() - t0, 2)
            result["n_features"] = len(names)
            result["feature_names"] = names

            t0 = time.time()
            fitted = models_mod.fit_model(
                proposal["model"]["family"], proposal["model"]["hyperparameters"],
                X_tr, self.y_train, X_val, self.y_val, seed)
            result["train_seconds"] = round(time.time() - t0, 2)
            result["model_info"] = fitted.info

            p_tr = fitted.predict_proba(X_tr)
            p_val = fitted.predict_proba(X_val)
            result["train_metrics"] = compute_metrics(self.y_train, p_tr)
            result["val_metrics"] = compute_metrics(self.y_val, p_val)
            result["overfit_gap"] = overfit_gap(result["train_metrics"], result["val_metrics"])

            # kept for future ensembling; validation predictions only
            np.save(out_dir / "val_predictions.npy", p_val.astype("float32"))
            write_json(out_dir / "metrics.json", {
                "train_metrics": result["train_metrics"],
                "val_metrics": result["val_metrics"],
                "overfit_gap": result["overfit_gap"],
                "feature_names": names,
                "model_info": fitted.info,
            })
        except ProposalError as exc:
            result.update(status="invalid_proposal", error=str(exc))
        except MemoryError:
            result.update(status="oom", error="MemoryError during training")
        except Exception as exc:  # noqa: BLE001 - any training failure is a logged outcome
            result.update(status="failed_train",
                          error=f"{type(exc).__name__}: {exc}")
            (out_dir / "traceback.txt").write_text(traceback.format_exc())

        result["runtime_seconds"] = round(time.time() - t_start, 2)
        return result
