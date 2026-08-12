from pathlib import Path

from fine_tuning_studio.resources import full_training_gate


def test_full_training_refuses_unsafe_vram(tmp_path: Path) -> None:
    gate = full_training_gate(1_000_000_000, 4.0, tmp_path)
    assert not gate.allowed
    assert "85%" in gate.reasons[0]


def test_small_full_training_fits(tmp_path: Path) -> None:
    assert full_training_gate(1_000_000, 4.0, tmp_path).allowed
