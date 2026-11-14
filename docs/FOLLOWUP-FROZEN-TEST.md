# 1.5B SFT continuation: frozen-test record

The successful Colab follow-up continued the previously trained Qwen2.5-1.5B-Instruct adapter on the existing SFT and pool groups. Epoch 3 was selected using the existing validation set: strict accuracy 108/200 (54%) and decision accuracy 173/200 (86.5%). The validation set was used for development and checkpoint selection.

The selected epoch-3 adapter was then evaluated on a newly generated frozen synthetic test: 400 groups, each rendered once with familiar and once with held-out wording (800 rows). The generator checked underlying policy/ticket cases against the original SFT, pool, validation and original test groups. Labels were balanced by action; half the groups use novel policy values. This is the same synthetic generator, so it does not establish real-world performance.

| Measure | Result |
|---|---:|
| Correct action | 674/800 (84.25%) |
| Exact evidence | 400/800 (50.0%) |
| Strict: action and exact evidence | 386/800 (48.25%) |
| Strict, familiar wording | 238/400 (59.5%) |
| Strict, held-out wording | 148/400 (37.0%) |

The group-bootstrap 95% interval for overall strict accuracy is approximately 44.38%–52.25%, describing test-sample uncertainty only. There is one training seed. The test has already been inspected and must not be reused as an untouched final test for subsequent tuning. For future claims, use a newly frozen test after selecting the new approach.

The test archive is `runs/followup-frozen-test/`: frozen cases, raw predictions, metrics and `FREEZE.json`. The source ZIP was `preference-lab-new-frozen-test-retry.zip`; an earlier attempt failed at adapter loading and produced no predictions. The test cases SHA-256 is `bf193eb69b33738e53ed21f33de44560b851397b93a0a1d819cc1e07b4f6c53f`. The selected adapter SHA-256 is `d5476835c038bac6bf58caed16b7a670baf3f45dfa576e023fcb33dd0e81e59b`; model revision is recorded in FREEZE.json. The adapter weights are not included in GitHub.

The 0.5B original study and 1.5B continuation differ in model, training and test data. Their scores cannot support a controlled attribution to any single change. DPO and GRPO were not rerun for this follow-up. Evidence alignment and all 800 grading flags were independently rechecked against the saved frozen cases and the project's verifier.
