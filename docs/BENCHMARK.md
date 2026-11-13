# Throughput benchmark (13 Nov 2026)

Machine: Apple M4, 16 GB, macOS 26.5, torch 2.15.0 on MPS (no CUDA). Python 3.12.10, transformers 5.23.0, trl 1.18.0, peft 0.21.3.
Model: Qwen/Qwen2.5-0.5B-Instruct, revision 7ae557604adf67be50417f59c2c2f167def9a775, float32.

Measured by `python -m preference_lab.doctor` (not a training run):
- Model load: 28.5 s (first load includes download).
- Generation: 20 greedy generations produced only 60 new tokens in total (the base model answered very briefly), 3.3 s, about 0.16 s per generation. Tokens per second from this run is not representative for longer outputs.
- LoRA training: 10 optimizer steps at batch 8, sequence length 58, 6.3 s, about 0.63 s per step.
- Memory: 2.0 GB allocated, 4.4 GB driver-allocated on MPS.

What this does and does not show: a 0.5B LoRA run fits comfortably on this machine. It says nothing yet about GRPO, which generates several completions per prompt at 90+ tokens each; that has to be benchmarked separately before any step budget is chosen.
