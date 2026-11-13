"""Supervised fine-tuning with LoRA and completion-only loss (the loss is computed on the answer tokens, not the prompt).

  python -m preference_lab.sft --name sft-overfit --split sft --limit 8 --epochs 30   # diagnostic only, excluded from comparisons
  python -m preference_lab.sft --name sft --split sft --epochs 1                      # the real run

Writes runs/<name>/{adapter/, config.json, train_log.jsonl, usage.json}.
"""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from .data import answer_json
from .evaluate import MODEL, REVISION, device, chat
from .splits import load

LORA = dict(r=8, lora_alpha=16, lora_dropout=0.0, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM")


def encode(tok, row, max_len=512):
    """Return input_ids and labels where only the answer (plus the end-of-turn token) is trained on."""
    prompt = tok(chat(tok, row), add_special_tokens=False)["input_ids"]
    answer = tok(answer_json(row["label"], row["required_ids"]) + "<|im_end|>", add_special_tokens=False)["input_ids"]
    ids = prompt + answer
    assert len(ids) <= max_len, f"{row['row_id']} is {len(ids)} tokens, over the {max_len} cap"
    return ids, [-100] * len(prompt) + answer


def collate(tok, items, dev):
    n = max(len(i) for i, _ in items)
    pad = tok.pad_token_id
    ids = torch.tensor([i + [pad] * (n - len(i)) for i, _ in items])
    lab = torch.tensor([l + [-100] * (n - len(l)) for _, l in items])
    att = torch.tensor([[1] * len(i) + [0] * (n - len(i)) for i, _ in items])
    return ids.to(dev), att.to(dev), lab.to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--split", default="sft"); ap.add_argument("--limit", type=int)
    ap.add_argument("--epochs", type=int, default=1); ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--micro", type=int, default=4); ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)
    dev = device()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.float32).to(dev)
    model = get_peft_model(model, LoraConfig(**LORA))
    rows = load(a.split)[: a.limit] if a.limit else load(a.split)
    data = [encode(tok, r) for r in rows]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr)
    before = torch.cat([p.detach().flatten().cpu() for p in params]).clone()
    out = Path("runs") / a.name; out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train_log.jsonl", "w")
    steps_per_epoch = max(1, len(data) // a.batch); total = steps_per_epoch * a.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0 if total < 20 else max(0.0, 1.0 - s / total))  # linear decay
    model.train(); t0 = time.time(); step = 0; tokens = 0
    for ep in range(a.epochs):
        order = list(range(len(data))); random.Random(a.seed + ep).shuffle(order)
        for b in range(steps_per_epoch):
            idx = order[b * a.batch:(b + 1) * a.batch]
            n_ans = sum(sum(1 for x in data[i][1] if x != -100) for i in idx)
            loss_sum = 0.0
            for m in range(0, len(idx), a.micro):
                ids, att, lab = collate(tok, [data[i] for i in idx[m:m + a.micro]], dev)
                # Memory fix: run the transformer, then apply the output head ONLY at answer positions. The full-vocabulary logits
                # for every prompt token (about 150k numbers each) were what used 12 GB. The loss is mathematically identical.
                inner = model.base_model.model
                hidden = inner.model(input_ids=ids, attention_mask=att).last_hidden_state[:, :-1]
                tgt = lab[:, 1:]
                sel = tgt != -100
                logits = inner.lm_head(hidden[sel])
                loss = torch.nn.functional.cross_entropy(logits, tgt[sel], reduction="sum") / n_ans
                loss.backward(); loss_sum += float(loss.detach()); tokens += int(att.sum())
            gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            opt.step(); sched.step(); opt.zero_grad(); step += 1
            log.write(json.dumps({"step": step, "epoch": ep, "loss": round(loss_sum, 5), "grad_norm": round(gn, 4), "lr": sched.get_last_lr()[0]}) + "\n"); log.flush()
            if step % 10 == 0 or step == 1: print(f"step {step}/{total} loss {loss_sum:.4f} grad_norm {gn:.3f}", flush=True)
    dt = time.time() - t0
    after = torch.cat([p.detach().flatten().cpu() for p in params])
    model.save_pretrained(out / "adapter")
    cfg = {"model": MODEL, "revision": REVISION, "lora": LORA, "lr": a.lr, "epochs": a.epochs, "batch": a.batch, "micro": a.micro, "seed": a.seed,
           "split": a.split, "n_examples": len(data), "limit": a.limit, "loss": "completion-only token cross-entropy", "schedule": "linear decay to 0 (constant if <20 steps)"}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    usage = {"optimizer_steps": step, "seconds": round(dt, 1), "tokens_seen": tokens, "device": dev, "trainable_params": int(sum(p.numel() for p in params)),
             "param_l2_change": float((after - before).norm()), "mps_peak_gb": round(torch.mps.driver_allocated_memory() / 1e9, 2) if dev == "mps" else None}
    (out / "usage.json").write_text(json.dumps(usage, indent=2))
    print(json.dumps(usage))


if __name__ == "__main__":
    main()
