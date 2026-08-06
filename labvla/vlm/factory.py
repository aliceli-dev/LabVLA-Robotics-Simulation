from __future__ import annotations

from .base import VLMBackend
from .mock_vlm import MockVLM


def build_vlm(backend: str = "mock") -> VLMBackend:
    name = backend.lower().strip()
    if name == "mock":
        return MockVLM()
    if name in {"api", "local", "local_smolvlm", "smolvlm"}:
        raise NotImplementedError(f"VLM backend '{backend}' is planned but not implemented yet")
    raise ValueError(f"Unknown VLM backend: {backend}")
