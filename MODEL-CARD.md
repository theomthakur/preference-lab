# Model card: Preference Lab checkpoints

**Base:** Qwen/Qwen2.5-0.5B-Instruct, revision `7ae557604adf67be50417f59c2c2f167def9a775`, float32, run on Apple M4 (MPS).

**Checkpoints (all LoRA adapters on the base, except GRPO which sits on the merged SFT model):**
| Name | Method | Trained on | Updates |
|---|---|---|---|
| SFT | supervised, completion-only loss, LoRA r=8 | sft split | 250 steps (5 epochs) |
| DPO clean | sigmoid DPO, beta 0.1, frozen SFT reference | 1,000 pool pairs | 125 steps |
| DPO corrupt | same, with 20% of pairs flipped | 1,000 pool pairs | 125 steps |
| GRPO | group-relative policy optimization, 4 samples per prompt, KL 0.04 | pool prompts, verifier reward | 32 updates |

**Intended use:** research and teaching on a small verifiable task. Not for real return decisions or any customer-facing use.

**Performance:** see RESULTS.md. The best model reaches about 12% strict correctness on the held-out test set, with the held-out phrasing template at about 8%.

**Known failure modes**
- Chooses ACCEPT or HOLD far more than the rules justify, depending on the checkpoint.
- Cites the right action with wrong evidence lines.
- Does not reliably fetch the ticket when tools are available.
- Generalizes poorly to new phrasing (the held-out template roughly halves accuracy).

**Not measured:** confidence or calibration. The model's text confidence and DPO margins are not calibrated probabilities, so no calibration claim is made.

**Safety and misuse:** no sensitive data, no deployment, no external API use. Verifier rewards are rule-based, so reward hacking is limited to the rule checker itself, which is tested.
