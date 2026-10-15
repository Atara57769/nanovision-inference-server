import hashlib
from pathlib import Path

import pytest

from app.model_management.checkpoint_download import CheckpointDownloadError, ensure_checkpoint


def test_existing_verified_checkpoint_is_reused_without_network(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"model weights")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    assert ensure_checkpoint(checkpoint, "https://example.invalid/model.pt", digest) == checkpoint


def test_existing_checkpoint_with_wrong_digest_is_refused(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"wrong model")

    with pytest.raises(CheckpointDownloadError, match="SHA-256 mismatch"):
        ensure_checkpoint(checkpoint, None, "0" * 64)
