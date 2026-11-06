"""Authenticated local state for single-use approvals and prompt ownership."""

import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import time
import tempfile

from .artifacts import create_json_exclusive, ignore_private_directory, protect_private_file
from .canonical import canonical_json


class ApprovalConsumedError(PermissionError):
    pass


class OwnershipRecordError(PermissionError):
    pass


class OperatorState:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self._key = _load_or_create_key(self.root / "hmac.key")

    def check_unused(self, approval_hash: str) -> None:
        if (self.root / "receipts" / f"{_approval_hash(approval_hash)}.json").exists():
            raise ApprovalConsumedError(f"approval hash already consumed: {approval_hash}")

    def consume_approval(
        self, approval_hash: str, *, run_directory: str | Path
    ) -> dict:
        value = _approval_hash(approval_hash)
        receipt = {
            "schema_version": 1,
            "approval_hash": value,
            "run_directory": str(Path(run_directory).expanduser().resolve()),
            "consumed_at_unix": time.time(),
        }
        try:
            create_json_exclusive(self.root / "receipts" / f"{value}.json", receipt, private=True)
        except FileExistsError as exc:
            raise ApprovalConsumedError(
                f"approval hash already consumed: {value}"
            ) from exc
        return receipt

    def sign_record(self, record: dict) -> dict:
        if not isinstance(record, dict):
            raise TypeError("ownership record must be a dictionary")
        canonical_record = json.loads(canonical_json(record))
        signature = hmac.new(
            self._key,
            canonical_json(canonical_record).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return {"record": canonical_record, "signature": signature}

    def verify_record(self, envelope: object) -> dict:
        if not isinstance(envelope, dict):
            raise OwnershipRecordError("ownership envelope must be a dictionary")
        record = envelope.get("record")
        signature = envelope.get("signature")
        if not isinstance(record, dict) or not _is_hash(signature):
            raise OwnershipRecordError("ownership envelope is malformed")
        expected = hmac.new(
            self._key,
            canonical_json(record).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(signature, expected):
            raise OwnershipRecordError("ownership record signature is invalid")
        return json.loads(canonical_json(record))


def _load_or_create_key(path: Path) -> bytes:
    if path.is_symlink():
        raise OwnershipRecordError("operator HMAC key cannot be a symlink")
    if path.exists():
        key = path.read_bytes()
        if len(key) != 32:
            raise OwnershipRecordError("operator HMAC key must contain exactly 32 bytes")
        protect_private_file(path)
    else:
        protect_private_file(path)
        with tempfile.TemporaryDirectory(dir=path.parent, prefix=".hmac-") as scratch:
            ignore_private_directory(Path(scratch))
            with tempfile.NamedTemporaryFile(dir=scratch, delete=False) as stream:
                temporary = Path(stream.name)
                key = secrets.token_bytes(32)
                stream.write(key)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                # Publish the complete key atomically, without replacing a winner.
                os.link(temporary, path)
            except FileExistsError:
                key = path.read_bytes()
    if len(key) != 32:
        raise OwnershipRecordError("operator HMAC key must contain exactly 32 bytes")
    return key


def _approval_hash(value: object) -> str:
    if not _is_hash(value):
        raise ValueError("approval hash must be 64 lowercase hexadecimal characters")
    return value


def _is_hash(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
