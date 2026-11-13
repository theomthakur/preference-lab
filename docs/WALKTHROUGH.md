# Interview walkthrough (say it out loud; about 5 minutes)

I built a small post-training experiment to learn the whole loop myself. The task is a fictional return-policy decision: given a policy and a ticket, answer ACCEPT, DENY or HOLD and cite the lines that decide it. I chose it because the right answer can be checked by a rule, so I could measure real progress without trusting a model's opinion.

First, the data. I generated 2,000 cases and split them by underlying case, so no case appears in two splits. I also wrote a second, independent version of the rules and checked that the two agree on all 1,800 rows I looked at. I hand-reviewed 40 examples before looking at the answers, and all 40 agreed.

Second, the training. I started with a small model, Qwen 0.5B. The first supervised run taught the format but collapsed to one answer. I didn't hide that. I ran a check that the training loop can overfit eight examples, then trained for five epochs, chosen on validation, and kept both results.

Third, preference training. I built pairs where the wrong answer is valid JSON but has the wrong action or the wrong evidence, so formatting cannot give it away. DPO used a frozen copy of the SFT model as its reference. I also flipped 20% of the labels as a control. Neither DPO run beat the SFT model beyond noise.

Fourth, GRPO. I merged the SFT weights first, so the frozen reference really is the SFT model. The speed test took about a minute for ten updates on my laptop. In 32 updates the reward stayed almost flat, and most groups of four samples had no difference in reward, so most updates carried little signal. I think that is a likely contributor, but I didn't test it, so I treat it as a hypothesis. I didn't fake a signal.

Fifth, the test. I froze the code and configs in a commit, then ran the held-out test once. Every trained model clearly beats the untrained base. None of the preference or RL steps beat SFT. Accuracy roughly halves on new wording, so the models partly memorize phrasing.

What I'd do next: a stronger starting model or more SFT data, so GRPO groups contain both passes and failures; several seeds before claiming anything; and a real tool-use training step, which I didn't reach.

Honest limits: one seed, a tiny model, synthetic labels from my own rules, and a laptop budget.
