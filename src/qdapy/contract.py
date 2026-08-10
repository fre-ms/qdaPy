"""The zotQDA exchange contract.

zotQDA and qdaZ write versioned files described by a machine-readable
contract.  Every file states its kind and version in its first column, so a
reader can recognise what it is holding without relying on the file name, and
can refuse a version it does not understand instead of quietly computing
something wrong.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

__all__ = ["contract", "example", "formats", "stamp_column"]


def _data_dir() -> Path:
    """Where the shipped contract and sample exports live.

    Resolved through ``importlib.resources`` rather than ``__file__`` so it
    also works from a zipped installation.
    """
    return Path(str(resources.files("qdapy") / "data"))


@lru_cache(maxsize=8)
def _load(path: str) -> dict[str, Any]:
    """Read and cache a JSON file.

    Cached because the contract is consulted on every read and never changes
    during a session; the key is the path, so a caller pointing at their own
    contract still gets that one.
    """
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def contract(path: str | Path | None = None) -> dict[str, Any]:
    """Return the exchange contract as a dictionary.

    Parameters
    ----------
    path:
        Optional path to an ``exchange-v*.json``.  Defaults to the copy
        shipped with this package, so nothing here needs a Zotero
        installation.
    """
    p = Path(path) if path is not None else _data_dir() / "exchange-v1.json"
    if not p.exists():
        raise FileNotFoundError(f"exchange contract not found: {p}")
    return _load(str(p))


def formats(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Return the supported formats, keyed by name."""
    return contract(path)["formats"]


def stamp_column(ct: dict[str, Any] | None = None) -> str:
    """The column in which every file declares its kind and version."""
    ct = ct if ct is not None else contract()
    return next(iter(ct["formats"].values()))["stampColumn"]


def example(name: str | None = None) -> Path | list[str]:
    """Path to a bundled reference file.

    The reference files from the contract are installed with this package, so
    every example and test runs without a Zotero installation.  They contain
    the awkward cases on purpose: quotes, the delimiter and a line break
    inside a field.

    Call without an argument to list what is available.  The list includes
    ``sample.qdpx``, a small REFI-QDA project for trying the .qdpx route.
    """
    d = _data_dir()
    if name is None:
        return sorted(p.name for p in d.iterdir()
                      if p.suffix in (".csv", ".qdpx"))
    p = d / name
    if not p.exists():
        raise FileNotFoundError(f"no such reference file: {name}")
    return p
