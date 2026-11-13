# Data card: return-policy decision set (generator v1.0)

**Synthetic.** Every policy, ticket and rule is invented. Labels come from a rule oracle (`oracle.py`), not from human judgment, so they are as good as the written rules, which are a simplified model of return policy and not legal or business advice.

## What a row is
One rendered prompt: a set of policy and ticket lines in shuffled order, with optional notes that must be ignored and sometimes a superseded window that must not be cited. The correct answer is an action (ACCEPT, DENY or HOLD) and the exact line ids that decide it.

## Size and splits (grouped by underlying case, so no case appears in two splits)
| Split | Groups | Rows | Use |
|---|---|---|---|
| sft | 400 | 400 | supervised fine-tuning |
| pool | 1,000 | 1,000 | preference pairs and GRPO prompts |
| val | 200 | 200 | every choice made before the test |
| test | 400 | 800 | held-out, used once |

Labels are balanced to within a few points in every split (about one third each). The test split has 400 extra rows with a held-out phrasing template and about half its groups use policy windows never seen in training.

## Known properties
- Each group has one or two renderings. A group never crosses splits. The audit checks shared group ids, shared case hashes and shared prompts across splits (all clean).
- Notes and superseded lines are included as distractors and are never required evidence.
- Hard negatives for preference pairs are valid JSON with wrong action, wrong or missing evidence, or made-up evidence, so formatting alone cannot separate them.
- Preference pair length differences average under one character, so length is not a shortcut.

## Known limits
- The oracle and the generator share the same assumptions, so a rule error would be consistent across the whole dataset.
- Phrasing is generated from templates. Real support tickets are messier.
- The 40-row hand review found no disagreement, but it is a small sample.

## Files and hashes
`data/manifest.json` records the generator version, seed and SHA-256 of each split file.
