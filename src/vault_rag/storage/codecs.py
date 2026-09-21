"""Binary codecs for normalized SQLite vector payloads."""

from __future__ import annotations

import math

import numpy as np  # pyright: ignore[reportMissingImports]
import numpy.typing as npt  # pyright: ignore[reportMissingImports]

from vault_rag.errors import StorageError

VECTOR_DTYPE = np.dtype("<f4")


def encode_vector(vector: np.ndarray, *, dimensions: int) -> bytes:
    """Validate, normalize, and encode one vector as little-endian float32."""
    if isinstance(dimensions, bool) or dimensions < 1:
        raise ValueError("vector dimensions must be positive")
    array = np.asarray(vector)
    if array.ndim != 1 or array.size != dimensions:
        raise ValueError("vector dimensions do not match its payload")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError("vector payload must be numeric")
    normalized = np.asarray(array, dtype=np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("vector must contain only finite values")
    norm = float(np.linalg.norm(normalized))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("vector must have non-zero finite norm")
    return np.asarray(normalized / norm, dtype=VECTOR_DTYPE).tobytes(order="C")


def decode_vector(raw: bytes, *, dimensions: int) -> npt.NDArray[np.float32]:
    """Decode one dimension-checked little-endian float32 vector."""
    if isinstance(dimensions, bool) or dimensions < 1:
        raise StorageError("stored vector dimensions are invalid")
    if len(raw) != dimensions * VECTOR_DTYPE.itemsize:
        raise StorageError("stored vector dimensions do not match its payload")
    vector: npt.NDArray[np.float32] = np.frombuffer(raw, dtype=VECTOR_DTYPE).astype(
        np.float32, copy=True
    )
    vector.setflags(write=False)
    return vector
