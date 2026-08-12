from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit


class OllamaError(RuntimeError):
    pass


@dataclass(frozen=True)
class OllamaModel:
    name: str
    size: int
    digest: str
    family: str
    parameter_size: str
    quantization_level: str


@dataclass(frozen=True)
class OllamaModelDetails:
    capabilities: tuple[str, ...]
    family: str
    parameter_size: str
    quantization_level: str


@dataclass(frozen=True)
class OllamaRunningModel:
    name: str
    size_vram: int
    context_length: int
    expires_at: str


class OllamaClient:
    def __init__(
        self,
        host: str | None = None,
        *,
        timeout: float = 3,
        stream_timeout: float = 300,
    ) -> None:
        self.host = _normalize_host(host or os.environ.get("OLLAMA_HOST", "127.0.0.1:11434"))
        self.timeout = timeout
        self.stream_timeout = stream_timeout

    def list_models(self) -> list[OllamaModel]:
        payload = self._request_json("/api/tags")
        return [
            OllamaModel(
                name=str(item.get("name") or item.get("model") or ""),
                size=int(item.get("size") or 0),
                digest=str(item.get("digest") or ""),
                family=str((item.get("details") or {}).get("family") or ""),
                parameter_size=str((item.get("details") or {}).get("parameter_size") or ""),
                quantization_level=str((item.get("details") or {}).get("quantization_level") or ""),
            )
            for item in payload.get("models", [])
            if isinstance(item, dict)
        ]

    def show_model(self, model: str) -> OllamaModelDetails:
        payload = self._request_json("/api/show", {"model": model})
        details = payload.get("details") or {}
        return OllamaModelDetails(
            capabilities=tuple(str(value) for value in payload.get("capabilities", [])),
            family=str(details.get("family") or ""),
            parameter_size=str(details.get("parameter_size") or ""),
            quantization_level=str(details.get("quantization_level") or ""),
        )

    def list_running_models(self) -> list[OllamaRunningModel]:
        payload = self._request_json("/api/ps")
        return [
            OllamaRunningModel(
                name=str(item.get("name") or item.get("model") or ""),
                size_vram=int(item.get("size_vram") or 0),
                context_length=int(item.get("context_length") or 0),
                expires_at=str(item.get("expires_at") or ""),
            )
            for item in payload.get("models", [])
            if isinstance(item, dict)
        ]

    def unload_model(self, model: str) -> None:
        self._request_json(
            "/api/generate",
            {"model": model, "keep_alive": 0, "stream": False},
        )

    def stream_chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        options: dict[str, int | float] | None = None,
        keep_alive: str | int = "5m",
    ) -> Iterator[dict[str, Any]]:
        request = self._request(
            "/api/chat",
            {
                "model": model,
                "messages": messages,
                "options": options or {},
                "keep_alive": keep_alive,
                "stream": True,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.stream_timeout) as response:
                for raw_line in response:
                    if not raw_line.strip():
                        continue
                    try:
                        chunk = json.loads(raw_line)
                    except json.JSONDecodeError as exc:
                        raise OllamaError("Ollama returned an invalid streaming response.") from exc
                    if not isinstance(chunk, dict):
                        raise OllamaError("Ollama returned an invalid streaming response.")
                    if error := chunk.get("error"):
                        raise OllamaError(f"Ollama generation failed: {str(error)[:500]}")
                    yield chunk
        except OllamaError:
            raise
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            raise OllamaError("The Ollama stream was interrupted or timed out.") from exc

    def _request_json(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        request = self._request(path, payload)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                value = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            raise OllamaError(f"Ollama rejected the request with HTTP {exc.code}.") from exc
        except (OSError, TimeoutError, urllib.error.URLError) as exc:
            raise OllamaError("Cannot connect to Ollama. Start Ollama and try again.") from exc
        except json.JSONDecodeError as exc:
            raise OllamaError("Ollama returned invalid JSON.") from exc
        if not isinstance(value, dict):
            raise OllamaError("Ollama returned an unexpected response.")
        if error := value.get("error"):
            raise OllamaError(f"Ollama request failed: {str(error)[:500]}")
        return value

    def _request(self, path: str, payload: dict[str, Any] | None = None) -> urllib.request.Request:
        data = json.dumps(payload).encode() if payload is not None else None
        return urllib.request.Request(
            f"{self.host}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST" if payload is not None else "GET",
        )


def _normalize_host(host: str) -> str:
    value = host.strip().rstrip("/")
    if "://" not in value:
        value = f"http://{value}"
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Ollama host must be an HTTP(S) origin without credentials or a path.")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("Ollama host contains an invalid port.") from exc
    return value
