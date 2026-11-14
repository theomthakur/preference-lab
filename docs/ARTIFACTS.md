# Artifacts: what is in GitHub, what is not, and what a fresh clone can run

**Later 1.5B SFT follow-up:** A separate 800-row synthetic frozen test and its saved predictions are documented in [the follow-up record](FOLLOWUP-FROZEN-TEST.md). Its adapter weights are not in this repository. Original-study details below remain historical.

## In GitHub
- **Code and tests:** `src/preference_lab/`, `tests/` (88 tests), `pyproject.toml`, `requirements.lock.txt`.
- **Dashboard (static, read-only):** `web/` (`index.html`, `app.js`, `style.css`, `favicon.svg`, `generate_data.py`, `README.md`, and `data.json`, a 1.2 MB derived copy of the held-out cases, predictions and metrics with a SHA-256 of each source), plus `vercel.json` and `.vercelignore`. `data.json` is derived and checked against the saved files; the saved files stay authoritative.
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
| `python -m pytest` | nothing extra | 88 tests pass |
| `python -m preference_lab.splits audit` | tracked data | leakage audit, prints `AUDIT: clean` |
| `python -m preference_lab.splits build` | nothing extra | regenerates `data/`; the files come out identical to the tracked ones (`git status` stays clean) |
| `python -m preference_lab.pairs` | tracked data | regenerates the preference pairs; identical to the tracked ones |
| `python -m preference_lab.demo` | tracked predictions | the saved-artifact demo; loads no model |
| `python -m preference_lab.test_analysis` | tracked predictions | recomputes the test metrics and prints them; writes nothing |
| `python -m preference_lab.test_analysis --check-against runs/test-results/test_metrics.json` | tracked predictions | confirms the saved metrics match the saved predictions |
| `python -m preference_lab.agent --scripted --name env-scripted-check` | tracked data | the 10-trajectory harness check; loads no model |
| `python3 web/generate_data.py --check` | tracked data and predictions (standard library only) | confirms `web/data.json` is byte-identical to a fresh build; writes nothing |
| `python3 -m http.server 4174 --bind 127.0.0.1 --directory web` | tracked files | serves the dashboard at http://127.0.0.1:4174 |

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

## Dashboard verification record (commit `9b4ccdd`, in a temporary clone of the committed repository)

**Verified**
- Pinned install from the lock file; 88 tests pass; `python3 web/generate_data.py --check` with system Python (standard library only) reports `data.json` byte-identical to a fresh build and writes nothing; the tracked tree stays clean. SHA-256 hashes of `runs/test-results/*` and `data/test.jsonl` were identical before and after all dashboard work.
- A real browser (Brave, which is Chromium, on macOS), driven by puppeteer-core, loading the page from a local server that applies `vercel.json`'s headers including the strict Content-Security-Policy. 43 checks, 0 failures, repeated against the fresh clone:
  - no console errors and no Content-Security-Policy violations;
  - bar values, interval geometry, counts and the zero-success note for all 3 wording views and 5 models, recomputed from the raw predictions, not from the saved metrics;
  - the paired table, the findings and the slide numbers match the data, and nothing claims DPO or GRPO improved on SFT;
  - all 15 filter combinations, and all 4,000 case-by-model renderings (evidence lines, check marks, saved output, verdict marks, reference answer, `aria-pressed`, same-case-other-wording);
  - keyboard: skip link; focus retained after Enter or Space on model tabs, case items, slide dots and at the end of the pager and the deck; arrow keys on the wording radios and slides; type-ahead in the filter select; a 3 px focus ring on nine kinds of control;
  - views: nav links, browser Back, direct and unknown hash links, page titles, focus moved to headings;
  - reduced motion (emulated): scroll calls use `auto`; without the preference they may be smooth;
  - mobile: selecting a case moves focus to it and scrolls it into view, and the back button returns focus to the list;
  - 320, 360, 390, 768, 1024, 1280 and 1440 px, each of the 3 views: no horizontal scrolling, no visible text under 12 px, every visible text element meeting WCAG AA contrast, every interactive target at least 44 px (links inside running text excluded);
  - failure states: `data.json` returning 404, and opening `index.html` from disk.
- Problems found by this testing and fixed: the presentation controls overflowed the deck on 320 and 360 px screens, and the footnote, intervals, focus and contrast issues listed in `docs/FINAL-STATUS.md`. One test failure was my test (on macOS an arrow key opens a closed select instead of changing it), so that check now uses type-ahead.

**Not verified**
- A real Vercel deployment. The Vercel command-line tool is not installed and the connector is not authorized. `vercel.json` was tested only as valid JSON and through a local server that applies its headers; Vercel itself was not run. The effect of `.vercelignore` was not tested, only its contents.
- Any browser other than Chromium (Firefox and Safari were not run).
- Screen readers (VoiceOver, NVDA, JAWS). ARIA attributes, status text and focus order were checked structurally, but nothing was heard.
- Real touch devices (mobile was emulated), browser zoom and text-spacing overrides, forced-colors mode, and the page without Google Fonts (the fallback fonts were not inspected visually).
- Contrast of graphics and control borders was checked for the design tokens, not element by element in every state (for example hover).
