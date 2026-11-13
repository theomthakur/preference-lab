"""Build the read-only dashboard data from committed synthetic cases and saved predictions."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ("base", "sft", "dpo_clean", "dpo_corrupt", "grpo")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    rows = read_jsonl(ROOT / "data/test.jsonl")
    ids = {row["row_id"] for row in rows}
    if len(ids) != len(rows):
        raise ValueError("Duplicate case IDs")
    predictions = {}
    for model in MODELS:
        values = read_jsonl(ROOT / f"runs/test-results/predictions_{model}.jsonl")
        by_id = {row["row_id"]: row for row in values}
        if len(by_id) != len(values) or set(by_id) != ids:
            raise ValueError(f"Missing, extra or duplicate predictions: {model}")
        predictions[model] = by_id
    metrics = json.loads((ROOT / "runs/test-results/test_metrics.json").read_text())
    cases = []
    for row in rows:
        case = {key: row[key] for key in ("row_id", "group_id", "rule", "prompt", "label", "required_ids", "template")}
        case["predictions"] = {
            model: {key: predictions[model][row["row_id"]].get(key) for key in ("output", "format_ok", "action_ok", "evidence_ok", "correct")}
            for model in MODELS
        }
        cases.append(case)
    for model in MODELS:
        actual = sum(bool(case["predictions"][model]["correct"]) for case in cases) / len(cases)
        if abs(actual - metrics["models"][model]["all_rows"]["point"]) > 0.000051:
            raise ValueError(f"Saved accuracy disagrees with predictions: {model}")
    (ROOT / "web/data.json").write_text(json.dumps({"cases": cases, "metrics": metrics}, separators=(",", ":")))
    print(f"Wrote {len(cases)} cases and {len(MODELS)} models")


if __name__ == "__main__":
    main()
