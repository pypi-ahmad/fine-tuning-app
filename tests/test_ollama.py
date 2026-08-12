import json
from collections.abc import Iterator
from typing import Any, cast
from urllib.error import URLError
from urllib.request import Request

import pytest

from fine_tuning_studio.ollama import OllamaClient, OllamaError


class FakeResponse:
    def __init__(self, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
        if isinstance(payload, list):
            self.lines = [json.dumps(item).encode() + b"\n" for item in payload]
            self.body = b"".join(self.lines)
        else:
            self.body = json.dumps(payload).encode()
            self.lines = [self.body]

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body

    def __iter__(self) -> Iterator[bytes]:
        return iter(self.lines)


def request_payload(request: Request) -> dict[str, Any]:
    return json.loads(cast(bytes, request.data or b"{}"))


def test_lists_installed_and_running_models(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = {
        "/api/tags": {
            "models": [
                {
                    "name": "gemma3:latest",
                    "size": 3_000,
                    "digest": "abc",
                    "details": {"parameter_size": "4B", "quantization_level": "Q4_K_M"},
                }
            ]
        },
        "/api/ps": {
            "models": [
                {
                    "name": "gemma3:latest",
                    "size_vram": 2_500,
                    "context_length": 4096,
                    "expires_at": "2026-08-12T12:00:00Z",
                }
            ]
        },
    }

    def fake_urlopen(request: Request, timeout: float) -> FakeResponse:
        assert timeout == 3
        path = request.full_url.removeprefix("http://127.0.0.1:11434")
        return FakeResponse(responses[path])

    monkeypatch.setattr("fine_tuning_studio.ollama.urllib.request.urlopen", fake_urlopen)
    client = OllamaClient()

    installed = client.list_models()
    running = client.list_running_models()

    assert installed[0].name == "gemma3:latest"
    assert installed[0].parameter_size == "4B"
    assert running[0].size_vram == 2_500
    assert running[0].context_length == 4096


def test_shows_model_capabilities(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: Request, timeout: float) -> FakeResponse:
        assert request.full_url.endswith("/api/show")
        assert request_payload(request) == {"model": "gemma3:latest"}
        return FakeResponse(
            {
                "capabilities": ["completion", "vision"],
                "details": {"family": "gemma3", "parameter_size": "4B"},
            }
        )

    monkeypatch.setattr("fine_tuning_studio.ollama.urllib.request.urlopen", fake_urlopen)

    details = OllamaClient().show_model("gemma3:latest")

    assert details.capabilities == ("completion", "vision")
    assert details.family == "gemma3"


def test_unload_uses_keep_alive_zero_without_deleting_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[Request] = []

    def fake_urlopen(request: Request, timeout: float) -> FakeResponse:
        seen.append(request)
        return FakeResponse({"done": True})

    monkeypatch.setattr("fine_tuning_studio.ollama.urllib.request.urlopen", fake_urlopen)

    OllamaClient().unload_model("gemma3:latest")

    assert seen[0].full_url.endswith("/api/generate")
    assert request_payload(seen[0]) == {
        "model": "gemma3:latest",
        "keep_alive": 0,
        "stream": False,
    }
    assert not any(part in seen[0].full_url for part in ("delete", "create", "pull"))


def test_streams_chat_ndjson_and_generation_options(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Request] = []

    def fake_urlopen(request: Request, timeout: float) -> FakeResponse:
        seen.append(request)
        assert timeout == 300
        return FakeResponse(
            [
                {"message": {"role": "assistant", "content": "Hel"}, "done": False},
                {
                    "message": {"role": "assistant", "content": "lo"},
                    "done": True,
                    "eval_count": 2,
                    "eval_duration": 1_000_000_000,
                },
            ]
        )

    monkeypatch.setattr("fine_tuning_studio.ollama.urllib.request.urlopen", fake_urlopen)
    chunks = list(
        OllamaClient().stream_chat(
            "gemma3:latest",
            [{"role": "user", "content": "Hi"}],
            options={"temperature": 0.5, "num_predict": 128},
            keep_alive="5m",
        )
    )

    assert "".join(chunk["message"]["content"] for chunk in chunks) == "Hello"
    assert request_payload(seen[0]) == {
        "model": "gemma3:latest",
        "messages": [{"role": "user", "content": "Hi"}],
        "options": {"temperature": 0.5, "num_predict": 128},
        "keep_alive": "5m",
        "stream": True,
    }


def test_reports_connection_and_stream_errors_without_raw_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "fine_tuning_studio.ollama.urllib.request.urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(URLError("secret endpoint detail")),
    )
    with pytest.raises(OllamaError, match="Cannot connect") as connection_error:
        OllamaClient().list_models()
    assert "secret endpoint detail" not in str(connection_error.value)

    monkeypatch.setattr(
        "fine_tuning_studio.ollama.urllib.request.urlopen",
        lambda *_args, **_kwargs: FakeResponse([{"error": "model failed"}]),
    )
    with pytest.raises(OllamaError, match="model failed"):
        list(OllamaClient().stream_chat("bad", [{"role": "user", "content": "Hi"}]))


@pytest.mark.parametrize(
    "host",
    ["file:///tmp/ollama", "http://user:password@localhost:11434", "http://localhost/x"],
)
def test_rejects_unsafe_ollama_hosts(host: str) -> None:
    with pytest.raises(ValueError, match="Ollama host"):
        OllamaClient(host)
