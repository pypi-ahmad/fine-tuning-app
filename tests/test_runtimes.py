from fine_tuning_studio.runtimes import runtime_verdict
from fine_tuning_studio.system_info import GPUInfo, MachineReport


def report(vendor: str, **torch: object) -> MachineReport:
    return MachineReport(
        os={"system": "Windows"},
        cpu={},
        memory={},
        disk={},
        gpus=[GPUInfo(vendor=vendor, name="test")],
        runtimes={"torch": torch},
        packages={"bitsandbytes": "1"},
    )


def test_cuda_ready_with_working_runtime() -> None:
    assert runtime_verdict(report("NVIDIA", cuda_available=True), "cuda").state == "ready"


def test_amd_rocm_is_unsupported_on_native_windows() -> None:
    verdict = runtime_verdict(report("AMD", hip_version=None), "rocm")
    assert verdict.state == "unsupported"
    assert "operating system" in verdict.reasons[0].lower()
