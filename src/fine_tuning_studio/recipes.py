from __future__ import annotations

import hashlib
import importlib.util
import inspect
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Recipe:
    id: str
    required_columns: tuple[str, ...]
    methods: tuple[str, ...]
    status: str = "stable"


RECIPES = {
    recipe.id: recipe
    for recipe in (
        Recipe("sft", ("text",), ("lora", "qlora", "full")),
        Recipe("continued_pretraining", ("text",), ("lora", "qlora", "full")),
        Recipe("dpo", ("prompt", "chosen", "rejected"), ("lora", "qlora")),
        Recipe("kto", ("prompt", "completion", "label"), ("lora", "qlora")),
        Recipe("reward", ("chosen", "rejected"), ("lora", "qlora", "full")),
        Recipe("orpo", ("prompt", "chosen", "rejected"), ("lora", "qlora"), "experimental"),
        Recipe("grpo", ("prompt",), ("lora", "qlora"), "experimental"),
    )
}


def validate_recipe(objective: str, method: str, columns: set[str]) -> list[str]:
    recipe = RECIPES[objective]
    errors = [
        f"Missing recipe column: {name}"
        for name in recipe.required_columns
        if name not in columns
    ]
    if method not in recipe.methods:
        errors.append(f"{objective.upper()} does not support {method.upper()}.")
    if objective == "orpo":
        errors.append("ORPO is experimental and unavailable in the installed TRL release.")
    return errors


def exact_reward(completions: list[str], ground_truth: list[str], **_: Any) -> list[float]:
    return [
        float(left.strip() == right.strip())
        for left, right in zip(completions, ground_truth, strict=True)
    ]


def numeric_reward(completions: list[str], ground_truth: list[str], **_: Any) -> list[float]:
    number = re.compile(r"[-+]?\d*\.?\d+")
    scores: list[float] = []
    for completion, truth in zip(completions, ground_truth, strict=True):
        left = number.findall(completion)
        right = number.findall(truth)
        scores.append(float(bool(left and right and float(left[-1]) == float(right[-1]))))
    return scores


def regex_reward(completions: list[str], ground_truth: list[str], **_: Any) -> list[float]:
    return [
        float(bool(re.search(pattern, value)))
        for value, pattern in zip(completions, ground_truth, strict=True)
    ]


def length_reward(completions: list[str], **_: Any) -> list[float]:
    return [min(len(value) / 200, 1.0) for value in completions]


REWARDS: dict[str, Callable[..., list[float]]] = {
    "exact": exact_reward,
    "numeric": numeric_reward,
    "regex": regex_reward,
    "length": length_reward,
}


def copy_trusted_reward(source: Path, job_directory: Path) -> tuple[Path, str]:
    source = source.resolve()
    if source.suffix != ".py" or not source.is_file():
        raise ValueError("Trusted reward module must be a Python file.")
    destination = job_directory / "inputs" / "trusted_reward.py"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location("fts_trusted_reward", destination)
    if not spec or not spec.loader:
        raise ValueError("Could not load trusted reward module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reward = getattr(module, "reward", None)
    if not callable(reward) or "completions" not in inspect.signature(reward).parameters:
        raise ValueError("Trusted module must define reward(completions, **kwargs).")
    probe = reward(completions=["probe"])
    if not isinstance(probe, list) or len(probe) != 1:
        raise ValueError("Trusted reward must return one score per completion.")
    return destination, digest


def load_trusted_reward(path: Path) -> Callable[..., list[float]]:
    spec = importlib.util.spec_from_file_location("fts_job_reward", path)
    if not spec or not spec.loader:
        raise ValueError("Could not load trusted reward module.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.reward
