"""Canonical plan integrity helpers for drift detection, not identity proof."""

import hashlib
import hmac
import json
import math


def canonical_json(value: object) -> str:
    """Serialize a plan deterministically, rejecting unsupported JSON values."""
    _validate_json_value(value, set())
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def approval_hash(plan: object) -> str:
    """Hash a canonical plan to detect changes after external human approval."""
    return hashlib.sha256(canonical_json(plan).encode("utf-8")).hexdigest()


def verify_approval(plan: object, expected_hash: object) -> bool:
    """Check plan integrity; human authorization remains an external gate."""
    if (
        not isinstance(expected_hash, str)
        or len(expected_hash) != 64
        or any(character not in "0123456789abcdef" for character in expected_hash)
    ):
        return False
    return hmac.compare_digest(approval_hash(plan), expected_hash)


def _validate_json_value(value: object, active: set[int]) -> None:
    value_type = type(value)
    if value is None or value_type in {bool, str, int}:
        return
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError("canonical JSON requires finite numbers")
        return
    if value_type is dict:
        _enter_container(value, active)
        for key, nested_value in value.items():
            if type(key) is not str:
                raise TypeError("canonical JSON mappings require string keys")
            _validate_json_value(nested_value, active)
        active.remove(id(value))
        return
    if value_type is list:
        _enter_container(value, active)
        for nested_value in value:
            _validate_json_value(nested_value, active)
        active.remove(id(value))
        return
    raise TypeError(f"unsupported canonical JSON type: {value_type.__name__}")


def _enter_container(value: object, active: set[int]) -> None:
    if id(value) in active:
        raise ValueError("canonical JSON cannot contain circular references")
    active.add(id(value))
