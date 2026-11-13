"""GRPO from the SFT model, with verifier rewards.

Reference model: the SFT adapter is MERGED into the base weights first (runs/sft-merged), so the frozen reference that TRL uses
is the SFT model. Only a new LoRA is trained on top, so the reference cannot silently be the untuned base.

  python -m preference_lab.grpo --name grpo-speed --max-steps 10          # speed test: time per update
  python -m preference_lab.grpo --name grpo --max-steps 32                # bounded real run (only after the speed test)

Reward: 1.0 only for a valid answer with the correct action AND exactly the required evidence ids; otherwise 0.0.
No length bonus, no format bonus.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import torch
from .evaluate import MODEL, REVISION, device, load_model
from .oracle import verify
from .splits import load
from .data import SYSTEM as ANSWER_SYSTEM

MERGED = Path("runs/sft-merged")
SFT_ADAPTER = Path("runs/sft/adapter")


def merge_sft():
    """Make sure the merged SFT model exists. The check is for the WEIGHTS file, not config.json: config and tokenizer files are
    tracked in git but the 1.9 GB weights are not, so a fresh clone has config.json and no weights."""
    if (MERGED / "model.safetensors").exists():
        return
    if not (SFT_ADAPTER / "adapter_model.safetensors").exists():
        raise FileNotFoundError(
            f"cannot build {MERGED}: the SFT adapter weights {SFT_ADAPTER}/adapter_model.safetensors are not present. "
            "Adapter weights are not stored in git. Retrain SFT first: python -m preference_lab.sft --name sft --epochs 5 --batch 8 --micro 2")
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.float32)
    m = PeftModel.from_pretrained(base, "runs/sft/adapter").merge_and_unload()
    MERGED.mkdir(parents=True, exist_ok=True)
    m.save_pretrained(MERGED); AutoTokenizer.from_pretrained(MODEL, revision=REVISION).save_pretrained(MERGED)


def to_dataset(rows):
    from datasets import Dataset
    recs = []
    for r in rows:
        recs.append({"prompt": [{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": r["prompt"]}],
                     "label": r["label"], "required_ids": r["required_ids"]})
    return Dataset.from_list(recs)


def reward(completions, label, required_ids, **_):
    out = []
    for comp, lab, req in zip(completions, label, required_ids):
        text = comp[0]["content"] if isinstance(comp, list) else comp
        out.append(1.0 if verify(text, lab, req)["correct"] else 0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--max-steps", type=int, default=10)
    ap.add_argument("--lr", type=float, default=5e-6); ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--gens", type=int, default=4); ap.add_argument("--prompts-per-step", type=int, default=2)
    ap.add_argument("--max-new", type=int, default=96); ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer
    merge_sft()
    rows = load("pool")
    ds = to_dataset(rows)
    out = Path("runs") / a.name; out.mkdir(parents=True, exist_ok=True)
    cfg = GRPOConfig(output_dir=str(out / "ckpt"), num_generations=a.gens, max_completion_length=a.max_new,
                     per_device_train_batch_size=a.gens * a.prompts_per_step, gradient_accumulation_steps=1,
                     learning_rate=a.lr, beta=a.beta, max_steps=a.max_steps, logging_steps=1, save_strategy="no",
                     report_to="none", seed=a.seed, bf16=False, fp16=False, remove_unused_columns=False,
                     model_init_kwargs={"dtype": torch.float32})
    trainer = GRPOTrainer(model=str(MERGED), reward_funcs=reward, args=cfg, train_dataset=ds,
                          peft_config=LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    t0 = time.time()
    res = trainer.train()
    dt = time.time() - t0
    trainer.model.save_pretrained(out / "adapter")  # the LoRA adapter on top of the merged SFT model
    log = [l for l in trainer.state.log_history if "loss" in l or "reward" in l]
    (out / "train_log.json").write_text(json.dumps(log, indent=1, default=str))
    usage = {"updates": a.max_steps, "seconds": round(dt, 1), "seconds_per_update": round(dt / max(a.max_steps, 1), 2),
             "completions_per_update": a.gens * a.prompts_per_step, "generations_total": a.gens * a.prompts_per_step * a.max_steps,
             "lr": a.lr, "beta": a.beta, "max_new": a.max_new, "reference": "merged SFT weights (frozen by TRL)", "device": device(),
             "mps_peak_gb": round(torch.mps.driver_allocated_memory() / 1e9, 2) if device() == "mps" else None}
    (out / "usage.json").write_text(json.dumps(usage, indent=2))
    print(json.dumps(usage, indent=2))


if __name__ == "__main__":
    main()
