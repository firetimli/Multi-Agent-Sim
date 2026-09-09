# Multi-agent AutoML for user behaviour prediction — Version 0

A minimal, closed agent loop for automatically building user-behaviour
prediction pipelines:

```
proposal -> train -> validation metrics -> critique -> revised proposal -> ...
```

Version 0 exists to prove the loop works, not to produce paper numbers. Two
agents (Modeling, Critic), one dataset (RetailRocket), spec-only proposals, a
deterministic controller, and a hard budget of 10 experiments.

## Quickstart

```bash
# 1. environment (Python 3.10+; the system python3 here is 3.7, so use a venv)
~/.local/share/mise/installs/python/3.12/bin/python3 -m venv .venv
.venv/bin/pip install --only-binary=:all: -r requirements.txt

# 2. smoke test: 3 experiments on synthetic data, mock backend, no downloads (~1s)
.venv/bin/python run.py --config configs/synthetic.yaml

# 3. RetailRocket: put events.csv at data/raw/retailrocket/events.csv, then
.venv/bin/python run.py --config configs/retailrocket.yaml --profile-only
.venv/bin/python run.py --config configs/retailrocket.yaml --proposal configs/example_proposal.json
.venv/bin/python run.py --config configs/retailrocket.yaml --backend codex_cli

# 4. results
.venv/bin/python run.py --leaderboard
```

RetailRocket data: the *Retailrocket recommender system dataset* on Kaggle
(`events.csv`, columns `timestamp,visitorid,event,itemid,transactionid`). Only
`events.csv` is used.

## Task formulation

User-level binary prediction, entirely config-driven so the label can change
later without touching code ([configs/retailrocket.yaml](configs/retailrocket.yaml)):

* a **cutoff** time `c` per split;
* one row per user with ≥ `min_history_events` in `[c - history_days, c)`;
* `label = 1` iff the user emits any `positive_events` in `(c, c + horizon_days]`.

Splits are three consecutive, non-overlapping label windows at the end of the
timeline (train / val / test), so validation is always a *future* period.
Default label is `transaction` within 7 days, from 14 days of history. Switch to
`positive_events: [addtocart, transaction]` for a denser engagement label.

## Trust boundary

| Trusted, human-controlled | Agent-controlled |
|---|---|
| dataset loading, labels, splits ([src/dataset.py](src/dataset.py)) | which features to select |
| feature implementations ([src/features.py](src/features.py)) | feature parameters |
| model implementations ([src/models.py](src/models.py)) | model family |
| metrics ([src/evaluator.py](src/evaluator.py)) | hyperparameters |
| execution + budget ([src/runner.py](src/runner.py), [src/store.py](src/store.py)) | the critique text |
| the loop ([src/orchestrator.py](src/orchestrator.py)) | — |

The agents have no filesystem or shell access to this repo: they receive a
rendered prompt string and return JSON, which trusted code validates and writes
under [generated/](generated/) and [runs/](runs/). "Agents cannot edit the
evaluator" is therefore structural, not a policy.

## Integrity properties already enforced

1. **Test labels are unreachable.** `Dataset.split("test")` raises. Nothing in
   the loop, the runner, or the metrics path can request them.
2. **No feature can see the future.** All features go through
   `UserEventIndex.history(user, cutoff, window_start)`, which asserts every
   returned event predates the row's cutoff.
3. **Fixed splits.** Cutoffs, windows, row counts and a manifest hash are
   recorded per run; the hash is stored on every experiment record.
4. **Catalog-bounded search.** Unknown features / families / hyperparameter
   names are rejected with an actionable message; out-of-range values are
   clamped and the clamp is reported back to the agent.
5. **Budget is a hard cap, and failures cost budget.** Otherwise an agent
   emitting garbage would get unlimited free retries. Schema *repair* rounds are
   free (they are not experiments); a proposal still invalid after repair is
   logged as `invalid_proposal` and consumes one experiment.
6. **Everything is logged.** One JSONL line per experiment plus every prompt and
   raw response on disk, with seeds, runtimes, proposal hash and split hash.
7. **Deterministic controller.** No LLM decides budget, stopping, or which model
   is best. Selection is `max validation AUROC`.

## Layout

```
configs/       retailrocket.yaml, synthetic.yaml (smoke), example_proposal.json
prompts/       modeling.md, critic.md          <- version-controlled, hashed per run
src/           dataset, features, models, evaluator, runner, store, orchestrator, utils
src/agents/    backend (codex_cli | mock), modeling_agent, critic_agent
generated/     <dataset>/<run_id>/<exp_id>/{proposal,metrics}.json, val_predictions.npy
runs/          experiments.jsonl  (source of truth, append-only)
               prompts/, responses/, run_configs/, summaries/
```

## Catalogs

Features: `history_count`, `unique_item_count`, `time_since_last_event`,
`mean_inter_event_time`, `recent_event_count(window_days)`, `item_repeat_ratio`,
`session_length(gap_minutes)`, `event_type_counts(normalize)`.

Models: `logistic_regression`, `random_forest`, `lightgbm`. No deep learning in
Version 0.

## Experiment record

One line per experiment in [runs/experiments.jsonl](runs/experiments.jsonl):

```json
{"run_id": "...", "experiment_id": "exp_001", "parent_experiment": null,
 "proposal": {"features": [...], "model": {...}}, "proposal_hash": "...",
 "train_metrics": {...}, "val_metrics": {"auroc": 0.74, "auprc": 0.2, "f1": 0.24},
 "overfit_gap": 0.03, "critique": {"diagnosis": ["overfitting"], ...},
 "seed": 42, "runtime_seconds": 104.9, "status": "success",
 "split_manifest_hash": "...", "artifacts": {...}}
```

`status ∈ {success, invalid_proposal, failed_train, oom}`.

## Codex backend

`agent.codex_cmd` in the config is the argv used to invoke the CLI (prompt on
stdin, completion on stdout); adjust it to match your installed Codex version.
`--backend mock` swaps in deterministic canned responses so the loop can be
exercised offline.

## Known Version-0 caveats

* LightGBM early-stops on the validation split, so validation AUROC is mildly
  optimistic. Fine while validation is only used for *relative* ranking; must be
  revisited before test evaluation.
* F1's threshold is chosen on the split being reported. When test evaluation is
  added, the threshold must come from validation.
* Users can appear in both train and val rows at different cutoffs (standard for
  temporal user-behaviour prediction). `train_val_user_overlap` is in the
  profile; a user-disjoint variant is a later option.
* The runner executes in-process. That is safe only because proposals are
  spec-only; a subprocess sandbox is required before free-form code generation.
* `configs/synthetic.yaml` is for plumbing tests only — never report its numbers.

## Deliberately not built yet

Data Agent, Feature Agent, Manager agent, the 8 fixed baselines, multiple
datasets, ensembling, test-set evaluation, free-form code generation,
single-agent vs. multi-agent comparison.
