"""DPO from the SFT checkpoint, with an explicit frozen reference.

  python -m preference_lab.dpo --name dpo-clean   --pairs data/pairs_clean.jsonl
  python -m preference_lab.dpo --name dpo-corrupt --pairs data/pairs_corrupt.jsonl

Policy  = base + the SFT adapter, trainable (the SFT weights are the starting point).
Reference = a SEPARATE copy of base + the SAME SFT adapter, frozen. This is NOT "base with the adapter turned off", which would be the
untuned base model and would make the DPO loss measure distance from the wrong model.
Loss (sigmoid DPO):  -log sigmoid( beta * [ (pi_c - ref_c) - (pi_r - ref_r) ] ), where pi_* and ref_* are summed answer-token log-probs.
"""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
import torch, torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from .evaluate import MODEL, REVISION, device, chat


def encode(tok, pair, answer, max_len=512):
    prompt = tok(chat(tok, pair), add_special_tokens=False)["input_ids"]
    ans = tok(answer + "<|im_end|>", add_special_tokens=False)["input_ids"]
    assert len(prompt) + len(ans) <= max_len
    return prompt + ans, len(prompt)


def seq_logp(model, ids, plen, dev):
    """Sum of log-probs of the answer tokens only (positions after the prompt)."""
    x = torch.tensor([ids], device=dev)
    inner = model.base_model.model
    hidden = inner.model(input_ids=x).last_hidden_state[:, :-1][:, plen - 1:]
    logits = inner.lm_head(hidden)
    tgt = x[:, plen:]
    return -F.cross_entropy(logits[0], tgt[0], reduction="sum")


def load_adapter_model(adapter, dev, trainable):
    base = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, dtype=torch.float32).to(dev)
    return PeftModel.from_pretrained(base, adapter, is_trainable=trainable)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--pairs", required=True); ap.add_argument("--sft", default="runs/sft/adapter")
    ap.add_argument("--beta", type=float, default=0.1); ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--epochs", type=int, default=1); ap.add_argument("--limit", type=int)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)
    dev = device()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    policy = load_adapter_model(a.sft, dev, True); policy.train()
    ref = load_adapter_model(a.sft, dev, False); ref.eval()
    for p in ref.parameters(): p.requires_grad_(False)
    pairs = [json.loads(l) for l in open(a.pairs)]
    if a.limit: pairs = pairs[: a.limit]
    data = []
    for p in pairs:
        c, pl = encode(tok, p, p["chosen"]); r, pl2 = encode(tok, p, p["rejected"]); assert pl == pl2
        data.append((c, r, pl))
    params = [p for p in policy.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr)
    # Reference check: at step 0 the policy and the frozen reference must give identical log-probs.
    with torch.no_grad():
        c, r, pl = data[0]
        d0 = abs(float(seq_logp(policy, c, pl, dev)) - float(seq_logp(ref, c, pl, dev)))
    assert d0 < 1e-3, f"policy and reference differ at step 0 by {d0}"
    out = Path("runs") / a.name; out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train_log.jsonl", "w")
    before = torch.cat([p.detach().flatten().cpu() for p in params]).clone()
    steps = (len(data) // a.batch) * a.epochs; step = 0; t0 = time.time(); toks = 0
    for ep in range(a.epochs):
        order = list(range(len(data))); random.Random(a.seed + ep).shuffle(order)
        for b in range(len(data) // a.batch):
            idx = order[b * a.batch:(b + 1) * a.batch]
            tot, marg, acc = 0.0, 0.0, 0
            for i in idx:
                c, r, pl = data[i]
                with torch.no_grad():
                    rc, rr = seq_logp(ref, c, pl, dev), seq_logp(ref, r, pl, dev)
                pc, pr = seq_logp(policy, c, pl, dev), seq_logp(policy, r, pl, dev)
                m = a.beta * ((pc - rc) - (pr - rr))
                loss = -F.logsigmoid(m) / len(idx)
                loss.backward(); tot += float(loss); marg += float(m) / len(idx); acc += int(float(m) > 0); toks += len(c) + len(r)
            gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0)); opt.step(); opt.zero_grad(); step += 1
            log.write(json.dumps({"step": step, "loss": round(tot, 5), "margin": round(marg, 5), "pref_acc": acc / len(idx), "grad_norm": round(gn, 4)}) + "\n"); log.flush()
            if step % 10 == 0 or step == 1: print(f"step {step}/{steps} loss {tot:.4f} margin {marg:.4f} acc {acc/len(idx):.2f} gn {gn:.2f}", flush=True)
    dt = time.time() - t0
    after = torch.cat([p.detach().flatten().cpu() for p in params])
    policy.save_pretrained(out / "adapter")
    (out / "config.json").write_text(json.dumps({"model": MODEL, "revision": REVISION, "init": a.sft, "reference": "frozen copy of the SFT adapter", "beta": a.beta, "lr": a.lr,
        "batch": a.batch, "epochs": a.epochs, "pairs": a.pairs, "n_pairs": len(data), "seed": a.seed, "loss": "sigmoid DPO", "step0_logp_diff": d0}, indent=2))
    (out / "usage.json").write_text(json.dumps({"optimizer_steps": step, "seconds": round(dt, 1), "tokens_seen": toks, "device": dev,
        "param_l2_change": float((after - before).norm()), "mps_peak_gb": round(torch.mps.driver_allocated_memory() / 1e9, 2) if dev == "mps" else None}))
    print(open(out / "usage.json").read())


if __name__ == "__main__":
    main()
