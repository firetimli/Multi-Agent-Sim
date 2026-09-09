"""Trusted model catalog: logistic_regression, random_forest, lightgbm.

TRUSTED INFRASTRUCTURE -- agents pick a family and propose hyperparameters.
Unknown families or unknown hyperparameter names are rejected (the agent gets a
repair round); out-of-range values are clamped and the clamp is reported back to
the agent so it learns the limits.

No deep learning in Version 0.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

from .utils import ProposalError

# lightgbm 3.x passes a kwarg sklearn 1.6+ renamed. Not actionable here, and it
# fires on every fit/predict. Filtered by exact message so nothing else is hidden.
warnings.filterwarnings("ignore", message=".*force_all_finite.*", category=FutureWarning)


@dataclass(frozen=True)
class HP:
    kind: str                    # "int" | "float" | "choice" | "bool"
    default: Any
    low: float | None = None
    high: float | None = None
    choices: tuple = ()
    allow_null: bool = False
    doc: str = ""


MODEL_CATALOG: dict[str, dict[str, HP]] = {
    "logistic_regression": {
        "C": HP("float", 1.0, 1e-4, 1e4, doc="inverse L2 regularisation strength"),
        "class_weight": HP("choice", None, choices=(None, "balanced"), allow_null=True),
        "max_iter": HP("int", 500, 100, 3000),
    },
    "random_forest": {
        "n_estimators": HP("int", 300, 50, 1000),
        "max_depth": HP("int", None, 2, 64, allow_null=True, doc="null = unlimited"),
        "min_samples_leaf": HP("int", 5, 1, 500),
        "max_features": HP("choice", "sqrt", choices=("sqrt", "log2", None), allow_null=True),
        "class_weight": HP("choice", None,
                           choices=(None, "balanced", "balanced_subsample"), allow_null=True),
    },
    "lightgbm": {
        "learning_rate": HP("float", 0.05, 0.005, 0.5),
        "num_leaves": HP("int", 31, 4, 512),
        "n_estimators": HP("int", 500, 50, 2000, doc="upper bound; early stopping on val"),
        "min_child_samples": HP("int", 20, 5, 500),
        "subsample": HP("float", 1.0, 0.3, 1.0, doc="row sampling; <1 enables bagging"),
        "colsample_bytree": HP("float", 1.0, 0.3, 1.0),
        "reg_alpha": HP("float", 0.0, 0.0, 20.0),
        "reg_lambda": HP("float", 0.0, 0.0, 20.0),
        "scale_pos_weight": HP("float", 1.0, 0.1, 500.0, doc="raise to counter class imbalance"),
        "max_depth": HP("int", -1, -1, 32, doc="-1 = unlimited"),
        "early_stopping_rounds": HP("int", 50, 10, 300, doc="0 disables early stopping"),
    },
}


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def validate_model_spec(model: Any) -> tuple[dict, list[str]]:
    if not isinstance(model, dict) or "family" not in model:
        raise ProposalError(
            "`model` must be an object with keys \"family\" and \"hyperparameters\". "
            f"Allowed families: {sorted(MODEL_CATALOG)}"
        )
    family = model["family"]
    if family not in MODEL_CATALOG:
        raise ProposalError(
            f"Unknown model family {family!r}. Version 0 supports only "
            f"{sorted(MODEL_CATALOG)} (no deep learning)."
        )
    spec = MODEL_CATALOG[family]
    raw = model.get("hyperparameters") or {}
    if not isinstance(raw, dict):
        raise ProposalError("`model.hyperparameters` must be an object.")
    unknown = set(raw) - set(spec)
    if unknown:
        raise ProposalError(
            f"Model {family!r} got unknown hyperparameter(s) {sorted(unknown)}; allowed: "
            f"{sorted(spec)}"
        )
    warnings: list[str] = []
    hp: dict[str, Any] = {}
    for name, hspec in spec.items():
        hp[name] = _coerce_hp(family, name, hspec, raw.get(name, hspec.default), warnings)
    return {"family": family, "hyperparameters": hp}, warnings


def _coerce_hp(family, name, spec: HP, value, warnings: list[str]):
    if value is None:
        if spec.allow_null:
            return None
        raise ProposalError(f"{family}.{name} may not be null.")
    if spec.kind == "choice":
        if value not in spec.choices:
            raise ProposalError(
                f"{family}.{name}={value!r} is not allowed; choices: {list(spec.choices)}"
            )
        return value
    if spec.kind == "bool":
        return bool(value)
    try:
        v = float(value) if spec.kind == "float" else int(value)
    except (TypeError, ValueError):
        raise ProposalError(f"{family}.{name} must be a {spec.kind}, got {value!r}.")
    lo, hi = spec.low, spec.high
    if lo is not None and v < lo:
        warnings.append(f"{family}.{name}={value} clamped up to {lo} (allowed range [{lo}, {hi}])")
        v = lo if spec.kind == "float" else int(lo)
    if hi is not None and v > hi:
        warnings.append(f"{family}.{name}={value} clamped down to {hi} (allowed range [{lo}, {hi}])")
        v = hi if spec.kind == "float" else int(hi)
    return v


def catalog_for_prompt() -> dict:
    out: dict[str, dict] = {}
    for family, spec in MODEL_CATALOG.items():
        out[family] = {}
        for name, h in spec.items():
            entry: dict[str, Any] = {"type": h.kind, "default": h.default}
            if h.kind == "choice":
                entry["choices"] = list(h.choices)
            elif h.low is not None:
                entry["range"] = [h.low, h.high]
            if h.allow_null:
                entry["nullable"] = True
            if h.doc:
                entry["doc"] = h.doc
            out[family][name] = entry
    return out


# --------------------------------------------------------------------------- #
# fitting
# --------------------------------------------------------------------------- #
class FittedModel:
    def __init__(self, estimator, info: dict):
        self.estimator = estimator
        self.info = info

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(X)[:, 1]


def fit_model(
    family: str, hp: dict, X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray, seed: int,
) -> FittedModel:
    if family == "logistic_regression":
        return _fit_logreg(hp, X_tr, y_tr, seed)
    if family == "random_forest":
        return _fit_rf(hp, X_tr, y_tr, seed)
    if family == "lightgbm":
        return _fit_lgbm(hp, X_tr, y_tr, X_val, y_val, seed)
    raise ProposalError(f"Unknown family {family!r} reached fit_model (catalog bug).")


def _fit_logreg(hp, X, y, seed) -> FittedModel:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    est = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(
            C=hp["C"], class_weight=hp["class_weight"], max_iter=hp["max_iter"],
            random_state=seed, n_jobs=None, solver="lbfgs")),
    ])
    est.fit(X, y)
    return FittedModel(est, {"n_iter": int(est["clf"].n_iter_[0])})


def _fit_rf(hp, X, y, seed) -> FittedModel:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    est = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=hp["n_estimators"], max_depth=hp["max_depth"],
            min_samples_leaf=hp["min_samples_leaf"], max_features=hp["max_features"],
            class_weight=hp["class_weight"], random_state=seed, n_jobs=-1)),
    ])
    est.fit(X, y)
    return FittedModel(est, {})


def _fit_lgbm(hp, X, y, X_val, y_val, seed) -> FittedModel:
    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover
        raise ProposalError(
            "lightgbm is not installed in this environment; propose "
            "logistic_regression or random_forest instead (pip install lightgbm)."
        ) from exc

    rounds = hp["early_stopping_rounds"]
    subsample = hp["subsample"]
    est = lgb.LGBMClassifier(
        learning_rate=hp["learning_rate"], num_leaves=hp["num_leaves"],
        n_estimators=hp["n_estimators"], min_child_samples=hp["min_child_samples"],
        subsample=subsample, subsample_freq=1 if subsample < 1.0 else 0,
        colsample_bytree=hp["colsample_bytree"], reg_alpha=hp["reg_alpha"],
        reg_lambda=hp["reg_lambda"], scale_pos_weight=hp["scale_pos_weight"],
        max_depth=hp["max_depth"], random_state=seed, n_jobs=-1, verbose=-1,
    )
    callbacks = [lgb.log_evaluation(0)]
    if rounds and rounds > 0:
        callbacks.append(lgb.early_stopping(int(rounds), verbose=False))
    est.fit(X, y, eval_set=[(X_val, y_val)], eval_metric="auc", callbacks=callbacks)
    return FittedModel(est, {
        "best_iteration": int(getattr(est, "best_iteration_", 0) or est.n_estimators),
        "early_stopped": bool(getattr(est, "best_iteration_", 0)
                              and est.best_iteration_ < hp["n_estimators"]),
    })
