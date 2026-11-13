# Preference Lab

A small, reproducible experiment in post-training a language model: synthetic data, a rule-based answer checker, supervised fine-tuning (SFT), preference tuning (DPO), and a shared evaluation. It runs on a laptop (Apple M4, 16 GB) with a 0.5B-parameter model.

**Status: work in progress.** Data, baseline, SFT and DPO are done and measured on a validation set. GRPO, the tool environment, the clean-versus-corrupted comparison write-up and the final held-out test are not done yet. Nothing here is a claim about frontier-scale training. All data is synthetic.

## The task

A support agent gets a fictional return policy and a ticket, and must answer with JSON:

```json
{"action": "DENY", "evidence": ["L3", "L5", "L6"]}
```

`action` is ACCEPT, DENY or HOLD (HOLD only when a needed fact is unknown). `evidence` lists the exact policy and ticket lines that decide it. Example: an opened accessory returned on day 10 that is not faulty is a DENY. If the fault status is missing, it is a HOLD.

A small rule oracle (`oracle.py`) decides the right answer. A score of "correct" needs the right action **and exactly the right evidence lines**, so a well-formatted wrong answer scores nothing.

## What has been measured so far (validation set, 200 cases)

| Model | Strictly correct | Right action | Exact evidence |
|---|---|---|---|
| Base (Qwen2.5-0.5B-Instruct) | 0 / 200 | 27.5% | 0% |
| SFT, 1 epoch | 0 / 200 | 35% | 1% |
| SFT, 5 epochs | 25 / 200 (12.5%) | 60% | 16.5% |
| DPO from the SFT model | 29 / 200 (14.5%) | 61.5% | 18% |

Read this carefully:
- The base model answers DENY about 80% of the time and never cites evidence.
- One epoch of SFT taught the format but collapsed to nearly always ACCEPT. Five epochs was chosen **after** seeing the validation result, so it is a validation-driven change to the original plan. Both runs are kept.
- DPO's gain over SFT is 4 cases out of 200 (11 wins, 7 losses on the same cases). That is within noise, so it is **not** shown to be better. It also shifted the model toward HOLD.
- The DPO run on 20% corrupted preference labels has trained but is not evaluated yet.
- The held-out test set has not been touched. These numbers are validation only, from one seed.

## How it is built

```
Scenario generator + independent rule oracle
        |
  grouped, leak-checked splits  (sft 400 / pool 1000 / val 200 / test 400 groups)
     |                 |
 SFT examples     preference pairs (clean, and 20% label-flipped copy)
     |                 |
 SFT model  ---->  DPO model (frozen SFT copy as reference)
     |
 same evaluator for every model
```

- Test cases include a held-out phrasing template and policy values never seen in training.
- Preference pairs use hard negatives (wrong action, missing, extra or made-up evidence), all valid JSON.
- The cross-check: a second, separately written oracle agrees with the first on every generated row.

## Run it

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e .                      # pinned versions are in requirements.lock.txt
python -m preference_lab.doctor          # hardware and speed check
python -m preference_lab.splits build    # data/ and a leakage audit
python -m preference_lab.pairs           # clean and corrupted preference pairs
python -m preference_lab.evaluate --split val --name base
python -m preference_lab.sft --name sft --epochs 5 --batch 8 --micro 2
python -m preference_lab.dpo --name dpo-clean --pairs data/pairs_clean.jsonl
python -m pytest                         # 35 tests
```

On a 16 GB Mac, set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4` to cap GPU memory. Each run writes its config, log, predictions (failures included) and measured usage to `runs/<name>/`.

## Honest limits

- Labels come from a rule oracle, not human judgment, and the generator and oracle share assumptions.
- One seed, a tiny model and a few hundred examples. Differences of a few cases are noise.
- Calibration and confidence are not measured.
- Not shown here: research results, multi-GPU training, or production use.

See `docs/BENCHMARK.md` for measured speed and memory.
