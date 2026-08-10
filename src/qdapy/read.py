"""Reading the exchange files, and checking them while reading."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from .contract import contract, stamp_column

__all__ = [
    "apply_mapping",
    "read",
    "read_codebook",
    "read_fragments",
    "read_history",
    "read_mapping",
    "read_uncoded",
]


class ContractError(ValueError):
    """A file does not match the exchange contract.

    Raised rather than returning something plausible: a reader that guesses
    at an unfamiliar file produces numbers nobody can trace back.
    """


def _sniff_delimiter(header: str) -> str:
    """Which delimiter the plugin used.

    The setting is the user's, so both occur in the wild.  Counting beats
    :class:`csv.Sniffer` here because a header line of plain field names has
    no quoting for the sniffer to work with.
    """
    if ";" in header and "," not in header:
        return ";"
    if ";" in header and header.count(";") > header.count(","):
        return ";"
    return ","


def read(  # noqa: C901 -- a chain of guard clauses, not a nested branch
    path: str | Path,
    fmt: str | None = None,
    *,
    strict: bool = True,
) -> pd.DataFrame:
    """Read any file zotQDA writes and check it against the contract.

    The file must declare a known kind, a version this package implements,
    and the columns the contract promises.  A file from a newer version is
    refused rather than guessed at.

    Files are UTF-8 with a byte-order mark and use ``,`` or ``;`` as the
    delimiter, depending on the setting in the plugin; both are detected.

    Parameters
    ----------
    path:
        The CSV to read.
    fmt:
        Optional expected format, e.g. ``"fragments"``.  When given, a file of
        a different kind raises -- useful in scripts that must not silently
        accept the wrong export.
    strict:
        When true (the default), a missing contract column raises: every
        column the contract declares is part of it, so an export without one
        is broken rather than merely different.  Extra columns are always
        accepted -- readers address columns by name and ignore what they do
        not know.

    Returns
    -------
    pandas.DataFrame
        With ``df.attrs`` holding ``qda_format``, ``qda_version`` and
        ``qda_grain``.
    """
    path = Path(path)
    ct = contract()
    stamp = stamp_column(ct)

    with open(path, encoding="utf-8-sig", newline="") as fh:
        first = fh.readline()
    if not first.strip():
        raise ContractError(f"empty file: {path}")
    sep = _sniff_delimiter(first)

    with open(path, encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter=sep))
    if not rows:
        raise ContractError(f"file has a header but no rows: {path}")

    columns = list(rows[0].keys())
    if stamp not in columns:
        raise ContractError(
            f"not a zotQDA exchange file (no {stamp!r} column): {path}"
        )

    ids = {r[stamp] for r in rows if r.get(stamp)}
    if len(ids) != 1:
        raise ContractError(f"file mixes formats: {', '.join(sorted(ids))}")
    declared = next(iter(ids))
    kind, _, version_text = declared.partition("/")

    if kind not in ct["formats"]:
        raise ContractError(
            f"unknown export kind {kind!r}; this package knows: "
            f"{', '.join(ct['formats'])}"
        )
    try:
        version = int(version_text)
    except ValueError as exc:
        raise ContractError(f"unreadable version in {declared!r}") from exc
    if version > ct["version"]:
        raise ContractError(
            f"file uses exchange version {version}, this package implements "
            f"{ct['version']} -- please update qdaPy"
        )
    if fmt is not None and fmt != kind:
        raise ContractError(f"expected a {fmt!r} export but got {kind!r}")

    spec = ct["formats"][kind]
    want = [c["key"] for c in spec["columns"]]
    missing = [c for c in want if c not in columns]
    if missing and strict:
        raise ContractError(
            f"{kind!r} export is missing contract columns: {', '.join(missing)}"
        )

    df = pd.DataFrame(rows, columns=columns).fillna("")
    for col in spec["columns"]:
        if col["type"] == "number" and col["key"] in df.columns:
            df[col["key"]] = pd.to_numeric(df[col["key"]], errors="coerce")

    df.attrs["qda_format"] = kind
    df.attrs["qda_version"] = version
    df.attrs["qda_grain"] = spec["grain"]
    return df


def read_fragments(path: str | Path) -> pd.DataFrame:
    """Read the coded fragments: one row per annotation and code."""
    return read(path, "fragments")


def read_uncoded(path: str | Path) -> pd.DataFrame:
    """Read the annotations no coder coded."""
    return read(path, "uncoded")


def read_codebook(path: str | Path) -> pd.DataFrame:
    """Read the code system: one row per code."""
    return read(path, "codebook")


def read_history(path: str | Path) -> pd.DataFrame:
    """Read the coding log: every ``add`` and ``remove``, oldest first."""
    return read(path, "history")


def read_mapping(path: str | Path) -> pd.DataFrame:
    """Read the consensus mapping: coder code to consensus code."""
    return read(path, "consensus-mapping")


def apply_mapping(
    fragments: pd.DataFrame,
    mapping: pd.DataFrame,
    *,
    coder_col: str = "codedBy",
) -> pd.DataFrame:
    """Add a ``consensusCode`` column without touching ``code``.

    The original coding stays visible next to its consensus interpretation.
    That separation is deliberate: rewriting the codings would inflate every
    agreement figure computed on the consensus system, by construction.
    """
    for col in ("coder", "coderCode", "consensusCode"):
        if col not in mapping.columns:
            raise KeyError(f"mapping is missing the column {col!r}")
    out = fragments.copy()
    lookup = {
        (str(coder), str(code)): str(consensus)
        for coder, code, consensus in zip(
            mapping["coder"], mapping["coderCode"], mapping["consensusCode"],
            strict=True,
        )
    }
    out["consensusCode"] = [
        lookup.get((str(c), str(code)), "")
        for c, code in zip(out[coder_col], out["code"], strict=True)
    ]
    return out
