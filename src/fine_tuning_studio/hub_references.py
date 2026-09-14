"""Parses and validates user-entered Hugging Face Hub references before any
network call is made with them, then plans/executes a download.

`normalize_hub_reference` is the trust boundary: everything downstream
(snapshot_download, disk-space checks) assumes it has already rejected
anything that isn't a plain "owner/name" repo ID. See preflight.py, which
separately queries the Hub for model metadata once a reference is chosen.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.utils import HFValidationError, validate_repo_id

RepoType = Literal["model", "dataset"]

_SUBPAGE_ROUTES = {"blob", "commits", "discussions", "raw", "resolve", "tree", "viewer"}


@dataclass(frozen=True)
class HubDownloadPlan:
    repo_id: str
    repo_type: RepoType
    revision: str
    commit_hash: str
    file_count: int
    total_bytes: int
    download_bytes: int
    cached_files: int


def normalize_hub_reference(value: str, expected_type: RepoType) -> str:
    reference = value.strip()
    if not reference:
        return ""
    if "://" not in reference:
        return _validated_repo_id(reference)

    # From here on `reference` is untrusted, user-typed text presumed to be a URL:
    # every check below narrows it before any part of it is used, and the function
    # must raise (never fall through) on anything it doesn't fully recognize.
    parsed = urlsplit(reference)
    if parsed.scheme.lower() != "https":
        raise ValueError("Hugging Face URLs must use HTTPS.")
    if parsed.username or parsed.password:
        raise ValueError("Hugging Face URLs must not contain credentials.")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Enter a valid huggingface.co URL.") from exc
    if parsed.hostname not in {"huggingface.co", "www.huggingface.co"} or port is not None:
        raise ValueError("Only huggingface.co repository URLs are supported.")

    path = parsed.path.strip("/")
    segments = path.split("/") if path else []
    if not segments or any(not segment for segment in segments):
        raise ValueError("Enter a valid huggingface.co repository URL.")
    if segments[0] == "spaces":
        raise ValueError("Hugging Face Spaces are not model or dataset repositories.")
    if segments[0] == "api":
        raise ValueError("Hugging Face API URLs are not repository URLs.")

    actual_type: RepoType = "dataset" if segments[0] == "datasets" else "model"
    if actual_type == "dataset":
        segments = segments[1:]
    if actual_type != expected_type:
        raise ValueError(f"This is a Hugging Face {actual_type} URL, not a {expected_type} URL.")

    # A repo ID is 1 segment (canonical, org-less) or 2 (org/name); anything after
    # that is a Hub subpage like /org/model/blob/main/file.bin. Distinguish the two
    # by checking whether the segment right after a 1- or 2-part ID is a known
    # subpage keyword, rather than assuming a fixed segment count.
    if len(segments) >= 4 and segments[2] in _SUBPAGE_ROUTES:
        route_index = 2
    elif len(segments) >= 3 and segments[1] in _SUBPAGE_ROUTES:
        route_index = 1
    else:
        route_index = len(segments)
    repo_parts = segments[:route_index]
    if len(repo_parts) not in {1, 2}:
        raise ValueError("Enter a valid Hugging Face repository URL.")
    return _validated_repo_id("/".join(repo_parts))


def plan_hub_download(repo_id: str, repo_type: RepoType, revision: str = "main") -> HubDownloadPlan:
    resolved_revision = revision.strip() or "main"
    result = snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=resolved_revision,
        dry_run=True,
    )
    files = result
    download_bytes = sum(file.file_size for file in files if file.will_download)
    # The Hub cache directory may not exist yet on a fresh machine; disk_usage needs
    # a real path, so walk up to the nearest existing ancestor to get a usable
    # (if slightly optimistic, since parent volumes can differ) free-space figure.
    available = shutil.disk_usage(_existing_cache_parent()).free
    if download_bytes > available:
        raise OSError(
            f"Download requires {download_bytes:,} bytes, but the Hugging Face cache disk "
            f"has only {available:,} bytes available."
        )
    return HubDownloadPlan(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=resolved_revision,
        commit_hash=files[0].commit_hash if files else resolved_revision,
        file_count=len(files),
        total_bytes=sum(file.file_size for file in files),
        download_bytes=download_bytes,
        cached_files=sum(not file.will_download for file in files),
    )


def download_hub_repository(plan: HubDownloadPlan) -> Path:
    result = snapshot_download(
        repo_id=plan.repo_id,
        repo_type=plan.repo_type,
        revision=plan.revision,
    )
    return Path(result)


def _validated_repo_id(value: str) -> str:
    try:
        validate_repo_id(value)
    except HFValidationError as exc:
        raise ValueError("Enter a valid Hugging Face repository ID.") from exc
    return value


def _existing_cache_parent() -> Path:
    path = Path(HF_HUB_CACHE).expanduser()
    while not path.exists() and path != path.parent:
        path = path.parent
    return path
