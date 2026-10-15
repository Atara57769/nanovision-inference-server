"""Fetch a configured public checkpoint safely when it is not cached locally."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


class CheckpointDownloadError(RuntimeError):
    """The configured checkpoint could not be downloaded or verified."""


def ensure_checkpoint(path: Path, url: str | None, expected_sha256: str | None) -> Path:
    """Return a verified checkpoint, downloading it atomically when needed."""

    if path.is_file():
        _verify_sha256(path, expected_sha256)
        return path
    if url is None or expected_sha256 is None:
        return path  # Model.load() reports the missing local-file configuration.

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".download")
    try:
        with urlopen(url, timeout=120) as response, temporary.open("wb") as target:
            while chunk := response.read(1024 * 1024):
                target.write(chunk)
        _verify_sha256(temporary, expected_sha256)
        os.replace(temporary, path)
    except (OSError, URLError) as error:
        raise CheckpointDownloadError(f"could not download checkpoint from {url}: {error}") from error
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _verify_sha256(path: Path, expected_sha256: str | None) -> None:
    if expected_sha256 is None:
        return
    with path.open("rb") as stream:
        actual_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual_sha256 != expected_sha256:
        raise CheckpointDownloadError(
            f"checkpoint SHA-256 mismatch for {path}: expected {expected_sha256}, got {actual_sha256}"
        )
