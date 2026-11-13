# Preference Lab results dashboard

This static, read-only site displays the committed held-out predictions and metrics. It never runs a model, changes evidence, or submits data. It is suitable for static hosting; the root `vercel.json` serves this directory.

To view locally from the repository root:

```sh
python3 -m http.server 4174 --bind 127.0.0.1 --directory web
```

Open http://127.0.0.1:4174. The checked-in `data.json` is a derived copy of `data/test.jsonl` and `runs/test-results/predictions_*.jsonl` plus `test_metrics.json`, joined by row ID. The source files remain authoritative. No model weights are included.

To regenerate the derived copy from saved source artifacts, run `python3 web/generate_data.py`. The script checks row IDs and accuracy against the saved metrics; it does not run models or rewrite the originals.
