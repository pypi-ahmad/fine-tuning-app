import pytest

from fine_tuning_studio.system_info import (
    GPUInfo,
    MachineReport,
    _hugging_face_token_present,
    qlora_capability,
)


def report(*, cuda: bool = True, free_vram: int = 8000) -> MachineReport:
    packages: dict[str, str | None] = {
        name: "1.0"
        for name in ("transformers", "datasets", "peft", "trl", "accelerate", "bitsandbytes")
    }
    return MachineReport(
        os={},
        cpu={},
        memory={},
        disk={},
        gpus=[GPUInfo(vendor="NVIDIA", name="Test GPU", memory_free_mb=free_vram)],
        runtimes={"torch": {"cuda_available": cuda}},
        packages=packages,
    )


def test_qlora_capability_ready() -> None:
    assert qlora_capability(report()) == (True, [])


def test_qlora_capability_reports_cuda_and_memory() -> None:
    ready, reasons = qlora_capability(report(cuda=False, free_vram=1000))
    assert not ready
    assert any("CUDA" in reason for reason in reasons)
    assert any("VRAM" in reason for reason in reasons)


@pytest.mark.parametrize("name", ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"])
def test_hugging_face_token_aliases_are_detected(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv(name, "test-token")

    assert _hugging_face_token_present()


def test_empty_hugging_face_token_variables_are_not_detected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "")

    assert not _hugging_face_token_present()
