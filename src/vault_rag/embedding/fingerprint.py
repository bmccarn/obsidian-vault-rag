"""Deterministic, route-independent embedding compatibility fingerprints."""

import hashlib
import json

from vault_rag.config import EmbeddingConfig

_NORMALIZATION_ALGORITHM = "l2-f32-v1"


def _digest(value: object) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def configuration_fingerprint(config: EmbeddingConfig, resolved_model: str) -> str:
    """Fingerprint resolved model semantics while excluding route and credentials."""
    if not isinstance(resolved_model, str) or not resolved_model.strip():
        raise ValueError("resolved model must be non-empty")
    return _digest(
        {
            "model": resolved_model,
            "endpoint_class": config.endpoint_class,
            "tokenizer": config.tokenizer,
            "target_min_tokens": config.target_min_tokens,
            "target_max_tokens": config.target_max_tokens,
            "overlap_tokens": config.overlap_tokens,
            "max_input_tokens": config.max_input_tokens,
            "revision": config.revision,
            "dimensions": config.dimensions,
            "normalization": _NORMALIZATION_ALGORITHM,
        }
    )


def observed_fingerprint(config_fingerprint: str, dimensions: int) -> str:
    """Add the provider-observed vector width to a configuration fingerprint."""
    if not config_fingerprint:
        raise ValueError("configuration fingerprint must be non-empty")
    if isinstance(dimensions, bool) or dimensions < 1:
        raise ValueError("dimensions must be positive")
    return _digest(
        {
            "configuration_fingerprint": config_fingerprint,
            "dimensions": dimensions,
        }
    )
