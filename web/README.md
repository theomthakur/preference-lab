# Preference Lab dashboard

A static, read-only site over the saved held-out results. It never runs a model, never changes a saved file, and original-study numbers come from `data.json` and follow-up numbers come from `followup.json`, so the text cannot drift from the results.

Three views, switched by the tabs (hash links): **Overview** (findings at a glance, the chart with 95% intervals and a seen vs held-out wording toggle, paired differences, and an explorer of the saved answers), **System design**, and **Presentation** (five slides, with left and right arrow keys).

## View it locally
```sh
python3 -m http.server 4174 --bind 127.0.0.1 --directory web
```
Open http://127.0.0.1:4174. Opening `index.html` directly from disk shows a message asking you to serve it over HTTP, because browsers block local file fetches.

## data.json is derived, and checked
`data.json` is built from `data/test.jsonl`, `runs/test-results/predictions_*.jsonl` and `runs/test-results/test_metrics.json`. Those sources stay authoritative.

```sh
python3 web/generate_data.py            # validate, then write web/data.json
python3 web/generate_data.py --check    # validate and confirm web/data.json is byte-identical to a fresh build (writes nothing)
```
The generator uses only the standard library and refuses to build if anything does not line up. For every model and every case it checks that:
- each case has exactly one prediction (no missing, extra or duplicate row ids);
- the prediction carries the same group, wording template, label, rule, split, novel-values flag and required evidence ids as the source case;
- the saved flags agree with each other;
- the accuracy recomputed from the predictions equals the saved metric exactly (all rows, seen wording, held-out wording), and the saved paired differences equal the recomputed ones.

It also records a SHA-256 of every source file inside `data.json`. A test (`tests/test_dashboard_data.py`) fails if the committed file is not byte-identical to a fresh build.

## What the page shows, and how
- **Intervals.** Each bar carries the saved 95% group-bootstrap interval. The axis is fixed across the three wording views so switching is a fair comparison. For zero successes the page explains that the bootstrap interval is always [0, 0] and shows an exact upper bound as supplementary arithmetic (it is not in the saved metrics).
- **Paired differences** come from the saved paired metrics. A difference is described as higher or lower only when its interval excludes zero.
- **Evidence lines** in a case are marked with a check mark, a left bar and hidden text for screen readers, so they are not distinguished by colour alone.

## Accessibility and behaviour decisions
- Keyboard focus is never replaced while it is held: selection and toggle state live in `aria-current` and `aria-pressed`, and end-of-list buttons use `aria-disabled`, not `disabled`.
- A skip link, a polite status region for changes, focus moved to the heading when a view or a case opens, and a back button on narrow screens.
- All text is at least 12 px; interactive targets are at least 44 px; colour pairs meet WCAG AA (tests check the tokens).
- No animation. Smooth scrolling is used only when the person has not asked for reduced motion.

## Deploying as a static site
The root `vercel.json` serves this folder with security headers, including a strict Content-Security-Policy (no inline scripts or styles). `.vercelignore` keeps sources and model weights out of a CLI upload. The headers were tested locally with a server that applies `vercel.json`; **a real Vercel deployment has not been made or verified.**

## Separate 1.5B follow-up
`followup.json` contains the saved metrics and freeze metadata from `runs/followup-frozen-test/`. The original `data.json` and comparison are unchanged. The displayed follow-up is a separate experiment, not a paired comparison with the original models.
