from importlib import import_module
from typing import Any

__all__ = ["ARCScorer"]

def __getattr__(name: str) -> Any:
    if name == "ARCScorer":
        # Lazy import to avoid importing submodule during package import
        return import_module(".scoring", __name__).ARCScorer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")