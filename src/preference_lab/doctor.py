"""Hardware and throughput check. Prints what this machine can actually do before any training is planned.

Run:  python -m preference_lab.doctor
Measures: model load time, 20 greedy generations, 10 LoRA optimizer steps, memory. Nothing is trained for real here.
"""
import json, platform, time, sys
import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

def main():
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    rev = HfApi().model_info(MODEL).sha
    print(json.dumps({"python": sys.version.split()[0], "torch": torch.__version__, "device": dev, "platform": platform.platform(), "model": MODEL, "revision": rev}))
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL, revision=rev)
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=rev, torch_dtype=torch.float32).to(dev)
    print(f"load_s={time.time()-t0:.1f}")
    msgs = [{"role": "user", "content": "A customer wants to return an opened accessory after 10 days. It is not faulty. Answer ACCEPT, DENY or HOLD."}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(dev)
    model.eval()
    t0 = time.time(); toks = 0
    with torch.no_grad():
        for _ in range(20):
            out = model.generate(**ids, max_new_tokens=64, do_sample=False)
            toks += out.shape[1] - ids["input_ids"].shape[1]
    g = time.time() - t0
    print(f"generation: 20 gens, {toks} new tokens, {g:.1f}s, {toks/g:.1f} tok/s, {g/20:.2f}s per gen")
    model.train()
    peft_model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    opt = torch.optim.AdamW([p for p in peft_model.parameters() if p.requires_grad], lr=2e-4)
    batch = tok([prompt + "DENY"] * 8, return_tensors="pt", padding=True).to(dev)
    labels = batch["input_ids"].clone()
    t0 = time.time(); losses = []
    for _ in range(10):
        loss = peft_model(**batch, labels=labels).loss
        loss.backward(); opt.step(); opt.zero_grad(); losses.append(float(loss))
    s = time.time() - t0
    print(f"train: 10 steps (batch 8, len {batch['input_ids'].shape[1]}), {s:.1f}s, {s/10:.2f}s per step, loss {losses[0]:.3f}->{losses[-1]:.3f}")
    if dev == "mps":
        print(f"mps_allocated_gb={torch.mps.current_allocated_memory()/1e9:.2f} driver_gb={torch.mps.driver_allocated_memory()/1e9:.2f}")

if __name__ == "__main__":
    main()
