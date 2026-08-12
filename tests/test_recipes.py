from pathlib import Path

import pytest

from fine_tuning_studio.evals import inspect_command
from fine_tuning_studio.recipes import REWARDS, copy_trusted_reward, validate_recipe


def test_dpo_contract_requires_preference_columns() -> None:
    errors = validate_recipe("dpo", "qlora", {"prompt", "chosen"})
    assert errors == ["Missing recipe column: rejected"]


def test_builtin_rewards_return_one_score_per_completion() -> None:
    assert REWARDS["exact"](["1"], ["1"]) == [1.0]
    assert REWARDS["numeric"](["answer 2"], ["2.0"]) == [1.0]
    assert len(REWARDS["length"](["hello"])) == 1


def test_trusted_reward_is_copied_hashed_and_probed(tmp_path: Path) -> None:
    source = tmp_path / "reward.py"
    source.write_text(
        "def reward(completions, **kwargs): return [1.0 for _ in completions]\n",
        encoding="utf-8",
    )
    destination, digest = copy_trusted_reward(source, tmp_path / "job")
    assert destination.exists()
    assert len(digest) == 64


def test_inspect_command_rejects_unknown_tasks(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown"):
        inspect_command(tmp_path, ["made-up"])
