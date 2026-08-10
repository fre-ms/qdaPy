"""Read a REFI-QDA project (.qdpx) into the tables the rest of qdaPy expects.

This is the way in for people who do not use zotQDA: MAXQDA, ATLAS.ti, NVivo,
QDA Miner and Dedoose all export .qdpx, and a .qdpx carries enough to run a
large part of this package.

It does not carry everything, and the difference is not a detail.  The
exchange CSVs zotQDA writes were designed for these analyses; .qdpx was
designed to move a project between programs.  What is missing is listed in
`QdpxProject.limitations`, printed as a warning on import, and set out in the
documentation next to what is possible.  Nothing here guesses at an absent
value: a column that cannot be filled is empty, and an analysis that needs it
fails rather than returning a flattering number.

Only the standard library is used - `zipfile` and `xml.etree` - so reading a
.qdpx adds no dependency to the package.
"""

from __future__ import annotations

import warnings
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import pandas as pd

from .contract import formats

__all__ = ["QdpxProject", "read_qdpx"]

NS = "{urn:QDA-XML:project:1.0}"

#: Project-level elements this reader does not turn into rows.  Every source
#: that is not text or PDF - pictures, audio, video - is counted the same way:
#: a project half of which is audio must not look like a complete one.
_UNSUPPORTED_ELEMENTS = ("Cases", "Variables", "Sets", "Notes", "Links", "Graphs")


@dataclass
class QdpxProject:
    """What a .qdpx yielded, and what it could not.

    The frames match the exchange contract column for column, so everything
    downstream works unchanged - but the columns a .qdpx cannot fill are
    present and empty.  `limitations` says which analyses that rules out.
    """

    fragments: pd.DataFrame
    codebook: pd.DataFrame
    history: pd.DataFrame
    uncoded: pd.DataFrame
    multi_coded: pd.DataFrame
    coders: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return (
            f"QdpxProject({len(self.fragments)} codings, "
            f"{len(self.codebook)} codes, {len(self.coders)} coders, "
            f"{len(self.sources)} sources)"
        )


# ----------------------------------------------------------------- reading --


CODEBOOK_COLUMNS = [
    "code", "codeId", "parent", "name", "level", "kind", "abbrev", "color",
    "memo", "nAnnotations", "nAnnotationsRecursive", "nDocumentsRecursive",
    "defined",
]

FRAGMENT_COLUMNS = [
    "code", "codeId", "citekey", "creator", "year", "title", "itemKey",
    "attachmentKey", "attachmentTitle", "pageLabel", "annotationKey",
    "annotationType", "color", "text", "comment", "weight", "allTags",
    "codedBy", "codedAt", "dateAdded", "dateModified", "positionKind",
    "positionStart", "positionEnd", "positionPage", "positionRects",
]


def _open_project(path: Path) -> tuple[ET.Element, dict[str, bytes]]:
    """Return the parsed project.qde and every internal source file.

    The spec puts project.qde at the root and the sources in a flat folder,
    but the folder is spelled `sources` by some tools and `Sources` by others,
    so the lookup is case-insensitive on the whole archive.
    """
    with zipfile.ZipFile(path) as zf:
        names = {n.lower(): n for n in zf.namelist()}
        qde = names.get("project.qde")
        if qde is None:
            msg = f"{path}: no project.qde in the archive - is this a .qdpx?"
            raise ValueError(msg)
        root = ET.fromstring(zf.read(qde))
        files = {
            n.rsplit("/", 1)[-1].lower(): zf.read(orig)
            for n, orig in names.items()
            if not n.endswith("/") and n != "project.qde"
        }
    return root, files


def _users(root: ET.Element) -> dict[str, str]:
    """GUID to name.  A coding whose user is unknown is attributed to its GUID."""
    out = {}
    for u in root.iter(f"{NS}User"):
        guid = u.get("guid", "")
        out[guid] = u.get("name") or guid
    return out


def _walk_codes(
    node: ET.Element, parent: str, inherited: str, out: list[dict[str, Any]]
) -> None:
    """Flatten the code tree into contract rows, depth first."""
    for code in node.findall(f"{NS}Code"):
        name = (code.get("name") or "").strip() or "(unnamed)"
        path = f"{parent}/{name}" if parent else name
        colour = code.get("color") or inherited
        memo = code.findtext(f"{NS}Description", "") or ""
        out.append(
            {
                "code": path,
                "codeId": code.get("guid", ""),
                "parent": parent,
                "name": name,
                "level": path.count("/") + 1,
                "kind": "code",
                "abbrev": "",
                "color": colour,
                "memo": memo.strip(),
                "defined": 1,
            }
        )
        _walk_codes(code, path, colour, out)


def _codebook(root: ET.Element) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    for book in root.iter(f"{NS}CodeBook"):
        for codes in book.findall(f"{NS}Codes"):
            _walk_codes(codes, "", "", rows)
    by_guid = {r["codeId"]: {"code": r["code"], "color": r["color"]} for r in rows}
    df = pd.DataFrame(rows, columns=CODEBOOK_COLUMNS)
    df.insert(0, "zotqdaFormat", "codebook/1")
    return df, by_guid


def _plain_text(src: ET.Element, files: dict[str, bytes]) -> str | None:
    """The source's plain text, from an internal file or inline.

    Returns None when the text lives outside the archive; selections then keep
    their positions but carry no text, which is honest and still enough for
    every positional measure.
    """
    inline = src.findtext(f"{NS}PlainTextContent")
    if inline is not None:
        return inline
    ref = src.get("plainTextPath") or ""
    name = ref.rsplit("/", 1)[-1].lower()
    blob = files.get(name)
    if blob is None:
        return None
    return blob.decode("utf-8-sig", errors="replace")


def _selection_base(src: ET.Element, sel: ET.Element, kind: str) -> dict[str, Any]:
    """The columns every row of one selection shares, whatever it is coded as."""
    base: dict[str, Any] = dict.fromkeys(FRAGMENT_COLUMNS, "")
    base["title"] = src.get("name") or ""
    base["itemKey"] = src.get("guid", "")
    base["attachmentKey"] = src.get("guid", "")
    base["attachmentTitle"] = src.get("name") or ""
    base["annotationKey"] = sel.get("guid", "")
    base["comment"] = (sel.findtext(f"{NS}Description", "") or "").strip()
    base["positionKind"] = kind
    return base


def _fill_text_position(
    base: dict[str, Any], sel: ET.Element, text: str | None
) -> None:
    """Character offsets, and the fragment itself when the text is at hand."""
    start, end = sel.get("startPosition"), sel.get("endPosition")
    base["positionStart"] = start or ""
    base["positionEnd"] = end or ""
    base["annotationType"] = "highlight"
    sliceable = text is not None and start is not None and end is not None
    base["text"] = text[int(start) : int(end)] if sliceable else (sel.get("name") or "")  # type: ignore[index,arg-type]


def _fill_pdf_position(base: dict[str, Any], sel: ET.Element) -> None:
    """A page and a rectangle - no continuum, so no unitizing measure."""
    base["positionPage"] = sel.get("page") or ""
    base["pageLabel"] = sel.get("page") or ""
    base["annotationType"] = "image"
    corners = ("firstX", "firstY", "secondX", "secondY")
    base["positionRects"] = " ".join(sel.get(k) or "" for k in corners).strip()
    base["text"] = sel.get("name") or ""


def _coding_row(
    base: dict[str, Any], coding: ET.Element,
    codes: dict[str, dict[str, str]], users: dict[str, str],
) -> tuple[dict[str, Any], str]:
    """One coding, plus the code path so the caller can build allTags."""
    ref = coding.find(f"{NS}CodeRef")
    guid = (ref.get("targetGUID") if ref is not None else "") or ""
    info = codes.get(guid, {"code": guid, "color": ""})
    user = coding.get("creatingUser") or ""
    stamp = coding.get("creationDateTime") or ""
    row = dict(base)
    row["code"] = info["code"]
    row["codeId"] = guid
    row["color"] = info["color"]
    row["codedBy"] = users.get(user, user)
    row["codedAt"] = stamp
    row["dateAdded"] = stamp
    row["dateModified"] = coding.get("modifiedDateTime") or stamp
    return row, info["code"]


def _selection_rows(
    src: ET.Element, sel: ET.Element, kind: str, text: str | None,
    codes: dict[str, dict[str, str]], users: dict[str, str],
) -> list[dict[str, Any]]:
    """One row per coding on this selection; one empty row if it has none."""
    base = _selection_base(src, sel, kind)
    if kind == "text":
        _fill_text_position(base, sel, text)
    else:
        _fill_pdf_position(base, sel)

    codings = sel.findall(f"{NS}Coding")
    if not codings:
        return [dict(base)]

    pairs = [_coding_row(base, c, codes, users) for c in codings]
    joined = "|".join(path for _, path in pairs)
    rows = [row for row, _ in pairs]
    for row in rows:
        row["allTags"] = joined
    return rows


def _source_rows(
    src: ET.Element, tag: str, files: dict[str, bytes],
    codes: dict[str, dict[str, str]], users: dict[str, str],
) -> list[dict[str, Any]]:
    """Every selection of one source, text and PDF alike."""
    text = _plain_text(src, files) if tag == "TextSource" else None
    rows: list[dict[str, Any]] = []
    for sel in src.findall(f"{NS}PlainTextSelection"):
        rows.extend(_selection_rows(src, sel, "text", text, codes, users))
    for sel in src.findall(f"{NS}PDFSelection"):
        rows.extend(_selection_rows(src, sel, "pdf", None, codes, users))
    return rows


def _sources(
    root: ET.Element, files: dict[str, bytes], codes: dict[str, dict[str, str]],
    users: dict[str, str],
) -> tuple[list[dict[str, Any]], list[str], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    names: list[str] = []
    skipped: dict[str, int] = {}
    for group in root.iter(f"{NS}Sources"):
        for src in group:
            tag = src.tag.replace(NS, "")
            if tag not in ("TextSource", "PDFSource"):
                skipped[tag] = skipped.get(tag, 0) + 1
                continue
            names.append(src.get("name") or src.get("guid") or "(unnamed source)")
            rows.extend(_source_rows(src, tag, files, codes, users))
    return rows, names, skipped


def _count_unsupported(root: ET.Element, skipped: dict[str, int]) -> None:
    for tag in _UNSUPPORTED_ELEMENTS:
        for group in root.iter(f"{NS}{tag}"):
            n = len(list(group))
            if n:
                skipped[tag] = skipped.get(tag, 0) + n


# ---------------------------------------------------------------- assembly --


def _coerce_numbers(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Give the columns the types the CSV reader gives them.

    Same names is not enough: `segments()` tests the positions with
    `is.finite`/numeric comparisons, and a column of strings quietly matches
    nothing.  The contract already says which columns are numbers, so this
    reads it rather than keeping a second list that could drift.
    """
    spec = formats().get(kind, {})
    numeric = [c["key"] for c in spec.get("columns", []) if c.get("type") == "number"]
    for name in [c for c in numeric if c in df.columns]:
        df[name] = pd.to_numeric(df[name], errors="coerce")
    return df


def _history_from(frag: pd.DataFrame) -> pd.DataFrame:
    """A coding log with additions only.

    REFI records when a coding was created and never that one was removed, so
    this is not the log zotQDA keeps.  Saturation and the timeline read it the
    same way; coder drift over withdrawn codings cannot be seen at all.
    """
    coded = frag[frag["code"] != ""]
    out = pd.DataFrame(
        {
            "zotqdaFormat": "history/1",
            "ts": coded["codedAt"],
            "user": coded["codedBy"],
            "action": "add",
            "code": coded["code"],
            "annotationKey": coded["annotationKey"],
            "citekey": "",
            "creator": "",
            "year": "",
            "title": coded["title"],
            "pageLabel": coded["pageLabel"],
            "text": coded["text"],
        }
    )
    return out[out["ts"] != ""].sort_values("ts").reset_index(drop=True)


def _multi_coded_from(frag: pd.DataFrame) -> pd.DataFrame:
    coded = frag[frag["code"] != ""]
    rows = []
    grouped = coded.groupby(["annotationKey", "codedBy"], sort=False)
    for (_key, coder), grp in grouped:
        distinct = sorted(set(grp["code"]))
        if len(distinct) > 1:
            rows.append(
                {
                    "zotqdaFormat": "multi-coded/1",
                    "document": grp["title"].iloc[0],
                    "text": grp["text"].iloc[0],
                    "codes": "+".join(distinct),
                    "n": len(distinct),
                    "coder": coder,
                }
            )
    cols = ["zotqdaFormat", "document", "text", "codes", "n", "coder"]
    return pd.DataFrame(rows, columns=cols)


def _limitations(frag: pd.DataFrame, uncoded: pd.DataFrame) -> list[str]:
    """What this project cannot answer, checked against what actually arrived."""
    out = [
        "no bibliographic metadata: citekey, creator and year are empty, so "
        "crosstabs by author, year or collection are not available",
        "no consensus exports: the three-phase consensus analyses need files "
        "only zotQDA writes",
        "no code abbreviations",
        "the coding log holds additions only: REFI records no removals, so "
        "code_drift over withdrawn codings cannot be computed",
    ]
    if uncoded.empty:
        out.append(
            "no uncoded segments: agreement therefore answers the friendlier "
            "question of how well the coders agreed about material one of them "
            "marked - pass uncoded= to units() only if you have that file"
        )
    if not (frag["codedBy"] != "").any():
        out.append(
            "no coders recorded: this export has no creatingUser on its "
            "codings, so no agreement coefficient can be computed at all"
        )
    if not (frag["codedAt"] != "").any():
        out.append(
            "no timestamps: saturation, the timeline and coder drift need "
            "creationDateTime, which this export does not carry"
        )
    if not (frag["positionStart"] != "").any():
        out.append(
            "no text positions: the unitizing measures and gamma need "
            "startPosition and endPosition on text selections"
        )
    return out


def read_qdpx(path: str | Path, *, warn: bool = True) -> QdpxProject:
    """Read a REFI-QDA project file into the tables qdaPy analyses.

    Parameters
    ----------
    path:
        A ``.qdpx`` archive written by any REFI-QDA conformant program.
    warn:
        Emit a :class:`UserWarning` listing what the file cannot support.
        Leave it on unless you have read the list once already.

    Returns
    -------
    QdpxProject
        Frames matching the exchange contract, plus what was skipped and why.

    Notes
    -----
    A .qdpx supports a subset of what the zotQDA CSV exports support.  See the
    documentation for the full comparison; `QdpxProject.limitations` states it
    for the file in your hand rather than in general.
    """
    p = Path(path)
    root, files = _open_project(p)
    users = _users(root)
    codebook, codes = _codebook(root)
    rows, sources, skipped = _sources(root, files, codes, users)
    _count_unsupported(root, skipped)

    frag = pd.DataFrame(rows, columns=FRAGMENT_COLUMNS)
    frag.insert(0, "zotqdaFormat", "fragments/1")
    frag = _coerce_numbers(frag, "fragments")
    codebook = _coerce_numbers(codebook, "codebook")
    uncoded = frag[frag["code"] == ""].reset_index(drop=True).copy()
    uncoded["zotqdaFormat"] = "uncoded/1"
    coded = frag[frag["code"] != ""].reset_index(drop=True)

    for df, kind in ((coded, "fragments"), (uncoded, "uncoded"),
                     (codebook, "codebook")):
        df.attrs["qda_format"] = kind
        df.attrs["qda_version"] = 1
        df.attrs["qda_origin"] = "qdpx"
        df.attrs["qda_source_file"] = str(p)

    project = QdpxProject(
        fragments=coded,
        codebook=codebook,
        history=_history_from(frag),
        uncoded=uncoded,
        multi_coded=_multi_coded_from(frag),
        coders=sorted({c for c in coded["codedBy"] if c}),
        sources=sources,
        skipped=skipped,
        limitations=_limitations(coded, uncoded),
    )
    if warn:
        _warn(project)
    return project


def _warn(project: QdpxProject) -> None:
    lines = [f"{len(project.fragments)} codings read from a REFI-QDA project."]
    if project.skipped:
        dropped = ", ".join(f"{k}: {v}" for k, v in sorted(project.skipped.items()))
        lines.append(f"Not represented in these tables - {dropped}.")
    lines.append("A .qdpx supports a subset of the analyses:")
    lines += [f"  - {x}" for x in project.limitations]
    lines.append("Pass warn=False once you have read this.")
    warnings.warn("\n".join(lines), UserWarning, stacklevel=3)
