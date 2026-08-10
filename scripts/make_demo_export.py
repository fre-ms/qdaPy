#!/usr/bin/env python3
"""Generate the demo export shared by qdaPy and qdaR.

A small synthetic interview study — "workload in nursing" — written as a
regular zotQDA exchange pair (zotqda-fragments-demo.csv and
zotqda-history-demo.csv), so everything that reads real exports reads it
too. The same bytes go into qdaPy's package data and qdaR's extdata; a
test in each package checks the two copies never drift apart.

The design, in the order the docs explain it:

* eight interviews from two settings, ``Station-1..4`` and
  ``Verwaltung-1..4``, whose theme profiles differ — that difference is
  what the correspondence-analysis map projects;
* nine codes in three themes (Belastung, Ressourcen, Bewältigung), with
  burden→coping partner pairs that tend to land on the same passage —
  that co-occurrence is what the dendrogram recovers;
* consensus segments (the three-phase method): both coders code the
  same annotations, so the units matrix lines up, with ~78 % category
  agreement, *systematic* confusions (a within-theme neighbour or the
  partner code, never uniform noise — that structure is what
  ``confusion()`` shows), a few segments bob left unrated, and a few
  double codings by ann (the ``multi_set_aside`` count);
* a coding history of ``add`` events in study order plus a handful of
  ``remove`` events, so the ``timeline()`` caveat has something to show.

Everything is derived from one seed and fixed timestamps: running this
script twice produces identical bytes, which is what the byte-identity
tests rely on. Regenerate with

    python3 scripts/make_demo_export.py

from the qdaPy checkout (qdaR must be checked out beside it).
"""

import csv
import hashlib
import io
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
QDAPY = HERE.parent
sys.path.insert(0, str(QDAPY / "src"))

from qdapy.contract import contract  # noqa: E402

QDAR = QDAPY.parent / "qdaR"
SEED = 20260808
T0 = np.datetime64("2026-06-08T09:00:00")   # fixed: no clock, ever

CODES = {   # code -> (weight on the ward, weight in administration, color)
    "Belastung/Zeitdruck":    (3.0, 1.2, "#ff6666"),
    "Belastung/emotional":    (2.6, 0.5, "#ff6666"),
    "Belastung/körperlich":   (1.8, 0.2, "#ff6666"),
    "Ressourcen/Team":        (2.2, 0.7, "#2ea8e5"),
    "Ressourcen/Autonomie":   (0.4, 2.4, "#2ea8e5"),
    "Ressourcen/Anerkennung": (0.7, 1.6, "#2ea8e5"),
    "Bewältigung/Humor":      (1.7, 0.6, "#5fb236"),
    "Bewältigung/Distanz":    (0.6, 1.9, "#5fb236"),
    "Bewältigung/Sport":      (0.5, 1.2, "#5fb236"),
}
# burden -> coping pairs that tend to be coded on the same passage
PARTNER = {"Belastung/emotional": "Bewältigung/Humor",
           "Belastung/Zeitdruck": "Bewältigung/Distanz",
           "Belastung/körperlich": "Bewältigung/Sport",
           "Ressourcen/Team": "Belastung/emotional"}
THEME = {c: c.split("/")[0] for c in CODES}

P_AGREE = 0.78      # bob picks ann's code
P_PARTNER = 0.45    # a partnered code joins the passage (via either coder)
P_SECOND = 0.08     # ann adds a second code herself -> multi_set_aside
P_SKIP = 0.07       # bob leaves the segment unrated
N_REMOVE = 6        # late second thoughts in the history


def code_id(code):
    # hashlib, not hash(): the built-in is salted per process and would
    # break the byte-identity the cross-package test checks
    return "c" + hashlib.md5(code.encode()).hexdigest()[:8]


def confusable(rng, code):
    """A systematic near-miss: same theme if possible, else the partner."""
    theme_mates = [c for c in CODES if THEME[c] == THEME[code] and c != code]
    pool = theme_mates + ([PARTNER[code]] if code in PARTNER else [])
    return str(rng.choice(pool)) if pool else code


def build(rng):
    fragments, events = [], []
    ann_no = 0
    for d, (doc, grp) in enumerate(
            [(f"Station-{i}", 0) for i in range(1, 5)]
            + [(f"Verwaltung-{i}", 1) for i in range(1, 5)]):
        weights = np.array([CODES[c][grp] for c in CODES], float)
        weights /= weights.sum()
        pos = 0
        for _ in range(int(rng.integers(26, 33))):
            ann_no += 1
            pos += int(rng.integers(20, 80))
            start, length = pos, int(rng.integers(60, 220))
            pos += length
            key = f"DEMO{ann_no:04d}"

            first = str(rng.choice(list(CODES), p=weights))
            codings = [("ann", first)]
            if rng.random() < P_SECOND:
                codings.append(("ann", confusable(rng, first)))
            if rng.random() >= P_SKIP:
                if rng.random() < P_AGREE:
                    codings.append(("bob", first))
                else:
                    codings.append(("bob", confusable(rng, first)))
            if first in PARTNER and rng.random() < P_PARTNER:
                who = "ann" if rng.random() < 0.5 else "bob"
                if ("ann", PARTNER[first]) not in codings:
                    codings.append((who, PARTNER[first]))

            for coder, code in codings:
                fragments.append({
                    "doc": doc, "d": d, "key": key, "coder": coder,
                    "code": code, "start": start, "end": start + length,
                })
    order = rng.permutation(len(fragments))
    for i, idx in enumerate(order):
        events.append((i, fragments[int(idx)]))
    return fragments, events


def iso(minutes):
    t = T0 + np.timedelta64(int(minutes), "m")
    return str(t) + ".000Z"


def rows_fragments(fragments):
    for f in fragments:
        added = iso(f["d"] * 720 + f["start"] // 10)
        yield {
            "zotqdaFormat": "fragments/1",
            "code": f["code"],
            "codeId": code_id(f["code"]),
            "citekey": f["doc"],
            "creator": "Demo",
            "year": "2026",
            "title": f"Interview {f['doc']}",
            "itemKey": f"ITEM{f['doc'][:4].upper()}{f['doc'][-1]}00",
            "attachmentKey": f"ATTA{f['doc'][:4].upper()}{f['doc'][-1]}00",
            "attachmentTitle": f"{f['doc']}.txt",
            "pageLabel": str(f["start"] // 1500 + 1),
            "annotationKey": f["key"],
            "annotationType": "highlight",
            "color": CODES[f["code"]][2],
            "text": f"Auszug {f['key'][-4:]}",
            "comment": "",
            "weight": "1",
            "allTags": f"#Code {f['code']}",
            "codedBy": f["coder"],
            "codedAt": added,
            "dateAdded": added,
            "dateModified": added,
            "positionKind": "text",
            "positionStart": str(f["start"]),
            "positionEnd": str(f["end"]),
            "positionPage": "",
            "positionRects": "",
        }


def rows_history(events, rng):
    rows = []
    for i, f in events:
        rows.append({
            "zotqdaFormat": "history/1",
            "ts": iso(i * 37),
            "user": f["coder"],
            "action": "add",
            "code": f["code"],
            "annotationKey": f["key"],
            "citekey": f["doc"],
            "creator": "Demo",
            "year": "2026",
            "title": f"Interview {f['doc']}",
            "pageLabel": str(f["start"] // 1500 + 1),
            "text": f"Auszug {f['key'][-4:]}",
        })
    # a few second thoughts, so remove events exist to be ignored correctly
    for j, idx in enumerate(rng.choice(len(rows) // 2, N_REMOVE,
                                       replace=False)):
        src = rows[int(idx)]
        rows.append(dict(src, ts=iso(len(rows) * 37 + j * 53),
                         action="remove"))
    rows.sort(key=lambda r: r["ts"])
    return rows


def write_csv(columns, rows):
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=columns, lineterminator="\r\n")
    w.writeheader()
    w.writerows(rows)
    return codecs_bom() + buf.getvalue().encode("utf-8")


def codecs_bom():
    return b"\xef\xbb\xbf"


def main():
    rng = np.random.default_rng(SEED)
    fragments, events = build(rng)
    ct = contract()
    files = {
        "zotqda-fragments-demo.csv": write_csv(
            [c["key"] for c in ct["formats"]["fragments"]["columns"]],
            rows_fragments(fragments)),
        "zotqda-history-demo.csv": write_csv(
            [c["key"] for c in ct["formats"]["history"]["columns"]],
            rows_history(events, rng)),
    }
    targets = [QDAPY / "src" / "qdapy" / "data"]
    if (QDAR / "inst" / "extdata").is_dir():
        targets.append(QDAR / "inst" / "extdata")
    else:
        print("WARNING: qdaR checkout not found, writing qdaPy copy only")
    for name, blob in files.items():
        for target in targets:
            (target / name).write_bytes(blob)
        print(f"{name}: {len(blob)} bytes -> {len(targets)} package(s)")


if __name__ == "__main__":
    main()
