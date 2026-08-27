"""Reading an easyQDA-CSV-Collection.

The collection (``easyqda-collection`` contract, EXCHANGE.md §6) is the
lossless, relational CSV serialisation of a project beside the REFI-QDA
``.qdpx``: thematic tables (``codes``, ``selections``, ``codings``,
``history`` …), one stamped CSV each, packaged as ``<project>.easyqda-csv.zip``
or an unpacked directory, with the binary sources carried alongside.

qdaR/qdaPy are only offers: this reader lets Python load every table into a
DataFrame, checked against the shipped contract, so an analysis can start from
plain CSV without the plugin or a ``.qdpx`` importer.  ``.qdpx`` stays the
interoperable default; the collection is the open, tool-agnostic archive.
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from .contract import _data_dir
from .read import ContractError, _sniff_delimiter

__all__ = [
    "collection_contract",
    "collection_tables",
    "read_collection",
    "read_collection_table",
]


@lru_cache(maxsize=4)
def _load(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def collection_contract(path: str | Path | None = None) -> dict[str, Any]:
    """The machine-readable ``easyqda-collection`` contract.

    Defaults to the copy shipped with this package, kept byte-identical with
    the one zotQDA generates, so nothing here needs a plugin installation.
    """
    p = Path(path) if path is not None else _data_dir() / "collection-v1.json"
    if not p.exists():
        raise FileNotFoundError(f"collection contract not found: {p}")
    return _load(str(p))


def collection_tables(ct: dict[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """The declared tables, keyed by name (``codes``, ``selections`` …)."""
    return (ct or collection_contract())["tables"]


def _read_csv_text(text: str, table: str, ct: dict[str, Any]) -> pd.DataFrame:
    """Parse one stamped collection table and check it against the contract."""
    spec = collection_tables(ct).get(table)
    if spec is None:
        raise ContractError(f"unknown collection table: {table!r}")
    stamp = spec["stampColumn"]

    first = text.splitlines()[0] if text.strip() else ""
    if not first:
        raise ContractError(f"empty table file: {table}")
    sep = _sniff_delimiter(first)
    rows = list(csv.DictReader(io.StringIO(text), delimiter=sep))
    if not rows:
        # a header with no data rows is valid for an (empty) table
        head = next(csv.reader(io.StringIO(first), delimiter=sep))
        return pd.DataFrame(columns=head)

    columns = list(rows[0].keys())
    if stamp not in columns:
        raise ContractError(f"not a collection table (no {stamp!r} column): {table}")

    ids = {r[stamp] for r in rows if r.get(stamp)}
    if len(ids) != 1:
        raise ContractError(f"table {table} mixes stamps: {', '.join(sorted(ids))}")
    declared = next(iter(ids))
    kind, _, version_text = declared.partition("/")
    if kind != spec["id"].split("/")[0]:
        raise ContractError(
            f"table {table} is stamped {kind!r}, expected "
            f"{spec['id'].split('/')[0]!r}")
    try:
        version = int(version_text)
    except ValueError as exc:
        raise ContractError(f"unreadable version in {declared!r}") from exc
    if version > ct["version"]:
        raise ContractError(
            f"collection uses version {version}, this package implements "
            f"{ct['version']} -- please update qdaPy")

    want = [c["key"] for c in spec["columns"]]
    missing = [c for c in want if c not in columns]
    if missing:
        raise ContractError(
            f"table {table} is missing contract columns: {', '.join(missing)}")

    df = pd.DataFrame(rows, columns=columns).fillna("")
    df.attrs["qda_table"] = table
    df.attrs["qda_version"] = version
    return df


def read_collection_table(
    path: str | Path,
    table: str | None = None,
    *,
    contract: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Read one collection table CSV and check it against the contract.

    ``table`` defaults to the file's base name (``codes.csv`` -> ``codes``).
    """
    p = Path(path)
    if table is None:
        table = p.stem
    ct = contract if contract is not None else collection_contract()
    return _read_csv_text(p.read_text(encoding="utf-8-sig"), table, ct)


def read_collection(
    src: str | Path,
    *,
    contract: dict[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    """Read a whole easyQDA-CSV-Collection into a dict of DataFrames.

    ``src`` is either a ``.easyqda-csv.zip`` file or an unpacked
    ``.easyqda-csv/`` directory (both hold ``tables/<name>.csv`` and a
    ``datapackage.json``).  Every declared table that is present is read and
    checked; missing optional tables are simply absent from the result.
    """
    ct = contract if contract is not None else collection_contract()
    tables: dict[str, pd.DataFrame] = {}
    p = Path(src)

    if zipfile.is_zipfile(p):
        with zipfile.ZipFile(p) as zf:
            present = {n for n in zf.namelist()}
            for name, spec in collection_tables(ct).items():
                entry = spec["file"]
                if entry in present:
                    text = zf.read(entry).decode("utf-8-sig")
                    tables[name] = _read_csv_text(text, name, ct)
    elif p.is_dir():
        for name, spec in collection_tables(ct).items():
            entry = p / spec["file"]
            if entry.exists():
                text = entry.read_text(encoding="utf-8-sig")
                tables[name] = _read_csv_text(text, name, ct)
    else:
        raise FileNotFoundError(
            f"not a collection (.easyqda-csv.zip file or directory): {src}")
    return tables
