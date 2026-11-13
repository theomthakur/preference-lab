# Artifacts: what is in GitHub, what is not, and what a fresh clone can run

## In GitHub
- **Code and tests:** `src/preference_lab/`, `tests/` (59 tests), `pyproject.toml`, `requirements.lock.txt`.
- **Documents:** README, RESULTS.md, DATA-CARD.md, MODEL-CARD.md, and `docs/`.
- **Data (generated, small):** `data/{sft,pool,val,test}.jsonl`, `data/pairs_clean.jsonl`, `data/pairs_corrupt.jsonl`, `data/corruption_flags.json` (kept apart from the learner data), and `data/manifest.json` with a SHA-256 for each split.
- **Run records** under `runs/<name>/`: configs, usage and timing, training logs, raw predictions (failures included) and metrics, for the base model, SFT (1 and 5 epochs), DPO (clean and corrupted), GRPO (speed test and the 32-update run) and the tool-environment episodes.
- **The one held-out test:** `runs/test-results/` holds `FREEZE.json`, the five prediction files and `test_metrics.json`.
- **Adapter descriptions** only: each adapter folder has `adapter_config.json` and a README, not the weights.
- **Tokenizer and config files** of the merged SFT model (`runs/sft-merged/`), not its weights.

## NOT in GitHub
Model weights are excluded by `.gitignore` (`*.safetensors`). They exist only on the machine that trained them.

| Local file | Size | What it is |
|---|---|---|
| `runs/sft/adapter/adapter_model.safetensors` | 17 MB | SFT adapter, 5 epochs (the one used downstream) |
| `runs/sft-1ep/adapter/adapter_model.safetensors` | about 17 MB | SFT adapter, 1 epoch (negative result) |
| `runs/dpo-clean/adapter/adapter_model.safetensors` | 17 MB | DPO on clean pairs |
| `runs/dpo-corrupt/adapter/adapter_model.safetensors` | 17 MB | DPO on 20% flipped pairs |
| `runs/grpo/adapter/adapter_model.safetensors` | 2.1 MB | GRPO adapter (sits on the merged SFT model) |
| `runs/sft-overfit*/adapter/adapter_model.safetensors` | about 17 MB each | the 8-example overfit diagnostics |
| `runs/sft-merged/model.safetensors` | 1.9 GB | SFT adapter merged into the base weights, used as GRPO's base and reference |
| Qwen2.5-0.5B-Instruct (revision `7ae5576...`) | 0.94 GB | the base model, downloaded from Hugging Face on first use |

**Consequence.** The trained models that produced the saved predictions cannot be loaded from a fresh clone. The saved predictions and metrics are the record of what they output. Retraining gives new weights from the same code and seeds, but this repo does not verify that another machine would reproduce the same numbers.

## What a fresh clone can run
Statuses were checked in a clean checkout of the committed repository (see the end of this file).

| Command | Needs | Result |
|---|---|---|
| `python -m pytest` | nothing extra | 59 tests pass |
| `python -m preference_lab.splits audit` | tracked data | leakage audit, prints `AUDIT: clean` |
| `python -m preference_lab.splits build` | nothing extra | regenerates `data/`; the files come out identical to the tracked ones (`git status` stays clean) |
| `python -m preference_lab.pairs` | tracked data | regenerates the preference pairs; identical to the tracked ones |
| `python -m preference_lab.demo` | tracked predictions | the saved-artifact demo; loads no model |
| `python -m preference_lab.test_analysis` | tracked predictions | recomputes the test metrics and prints them; writes nothing |
| `python -m preference_lab.test_analysis --check-against runs/test-results/test_metrics.json` | tracked predictions | confirms the saved metrics match the saved predictions |
| `python -m preference_lab.agent --scripted --name env-scripted-check` | tracked data | the 10-trajectory harness check; loads no model |

## What needs a Hugging Face download but no training
- `python -m preference_lab.doctor` (about 0.94 GB on first use, then cached).
- `python -m preference_lab.evaluate --split val --name base-check` (base model only). Use a new `--name`, because the default folder names are already tracked.

## What needs retraining first
Retraining reproduces the code path; it is not a promise of identical numbers. Times were measured on an Apple M4 with 16 GB.

| Step | Command | Measured |
|---|---|---|
| SFT, 5 epochs | `python -m preference_lab.sft --name sft --epochs 5 --batch 8 --micro 2` | 641 s |
| DPO, clean | `python -m preference_lab.dpo --name dpo-clean --pairs data/pairs_clean.jsonl` | 788 s |
| DPO, corrupted | `python -m preference_lab.dpo --name dpo-corrupt --pairs data/pairs_corrupt.jsonl` | 813 s |
| GRPO speed test | `python -m preference_lab.grpo --name grpo-speed --max-steps 10` | 65 s (also builds the 1.9 GB merged model) |
| GRPO, 32 updates | `python -m preference_lab.grpo --name grpo --max-steps 32` | 204 s |

Evaluating any trained model (SFT, DPO, GRPO) on validation, and running the tool environment with an adapter, also requires these weights. GRPO fails fast with a clear message if the SFT adapter is missing; it will not run on config files alone.

## What cannot be rerun
- `python -m preference_lab.final_test` refuses to run: `runs/test-results/` exists and the held-out test has been used. Do not tune against it. A new model should be judged on a newly generated validation set and a separately frozen new test.

## Verification record (review pass)
Checked on commit `f2c7392` in a temporary clone of the committed repository: lock-file installs (uv and pip) matched the 53 pinned packages; 59 tests passed; the audit was clean; `splits build` and `pairs` reproduced the tracked data byte for byte; `demo`, `test_analysis --check-against`, `agent --scripted` reproduced the tracked outputs; `grpo` failed fast without weights; `final_test` refused to run. Everything marked "can run" above was exercised this way.
