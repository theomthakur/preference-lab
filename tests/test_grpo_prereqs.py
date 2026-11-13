"""GRPO must not run on a half-present merged model (config files without weights), which is what a fresh clone contains."""
import pytest
from preference_lab import grpo


def test_missing_weights_fail_fast_with_a_clear_message(tmp_path, monkeypatch):
    merged = tmp_path / "sft-merged"; merged.mkdir()
    (merged / "config.json").write_text("{}")                      # config present, weights absent: the fresh-clone state
    monkeypatch.setattr(grpo, "MERGED", merged)
    monkeypatch.setattr(grpo, "SFT_ADAPTER", tmp_path / "no-adapter")
    with pytest.raises(FileNotFoundError, match="Retrain SFT first"):
        grpo.merge_sft()


def test_existing_merged_weights_are_reused(tmp_path, monkeypatch):
    merged = tmp_path / "sft-merged"; merged.mkdir()
    (merged / "model.safetensors").write_bytes(b"x")
    monkeypatch.setattr(grpo, "MERGED", merged)
    monkeypatch.setattr(grpo, "SFT_ADAPTER", tmp_path / "no-adapter")
    assert grpo.merge_sft() is None                                # returns without needing the adapter
