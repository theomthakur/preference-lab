# Preference Lab

**Live dashboard:** [preference-lab.vercel.app](https://preference-lab.vercel.app/)

Preference Lab is a small experiment in teaching a language model to make decisions and explain them. Given a fictional store policy and a return request, the model must answer **accept, deny or hold** and cite the exact lines supporting its decision.

The project explores supervised fine-tuning (SFT), preference tuning (DPO), reinforcement learning (GRPO), and how to evaluate their results honestly. The dashboard lets you explore saved answers without running a model.

## What happened

The original study compared versions of a 0.5B model. SFT improved on the untrained model, but DPO and GRPO did not show a clear gain over SFT. The strongest observed strict score was about 12% on its held-out synthetic test.

A later, separate experiment continued SFT on a 1.5B model. After selecting the checkpoint using validation data, we evaluated it on a newly generated frozen test:

- **84.25%** correct decisions (674 of 800 answers).
- **50%** exact evidence (400 of 800).
- **48.25%** correct decisions *and* exact evidence (386 of 800).
- Strict accuracy was **59.5%** on familiar wording and **37%** on held-out wording.

The two tests are different: these figures are not a controlled comparison of the models. Both experiments use synthetic cases from the same rule-based generator, with one training seed each. The follow-up is useful evidence of progress, not a production-readiness claim. See [RESULTS.md](RESULTS.md) for details and [the follow-up record](docs/FOLLOWUP-FROZEN-TEST.md) for provenance.

## Explore the dashboard

The site is read-only. It shows the later result separately from the original model comparison and its saved examples; it never runs a model.

```bash
python3 -m http.server 4174 --bind 127.0.0.1 --directory web
```

Open [http://127.0.0.1:4174](http://127.0.0.1:4174). The root `vercel.json` is configured to serve `web/` as a static site. A live Vercel deployment has not been verified.

## Explore the code

- `data/` — synthetic training, validation and original test data.
- `src/preference_lab/` — generator, rule checker, training and evaluation.
- `runs/test-results/` — original frozen-test evidence.
- `runs/followup-frozen-test/` — new 1.5B frozen cases, predictions, metrics and provenance.
- `web/` — static dashboard and derived data.
- `docs/ARTIFACTS.md` — what is saved and what a fresh clone can run.

The trained adapters and model weights are not in GitHub; saved predictions and metrics can be explored without them. The follow-up adapter remains in the separately downloaded experiment ZIP.

## Run the research code

The original environment was verified on macOS with Python 3.12. Install the pinned dependencies in a fresh environment:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip sync requirements.lock.txt
uv pip install -e . --no-deps
python -m pytest
python -m preference_lab.demo
python3 web/generate_data.py --check
```

The lock file was produced for macOS arm64 and is not hash-pinned; Linux/CUDA requires a separately validated environment. Running training or inference may download model weights and consume GPU resources. The original experiments, additional limitations and exact setup details are in [the artifact guide](docs/ARTIFACTS.md), [model card](MODEL-CARD.md) and [data card](DATA-CARD.md).
