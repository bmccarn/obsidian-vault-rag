"""OpenAI-compatible embedding client and compatibility fingerprints."""

from .client import EmbeddingBatch, EmbeddingClient, EmbeddingFailure, ModelProbe
from .fingerprint import configuration_fingerprint, observed_fingerprint

__all__ = [
    "EmbeddingBatch",
    "EmbeddingClient",
    "EmbeddingFailure",
    "ModelProbe",
    "configuration_fingerprint",
    "observed_fingerprint",
]
