#!/usr/bin/env python3
"""Build shared/references.bib from the Zotero library.

Reads every work on docs/en/dev/literature.md, looks it up in the running
Zotero via the Better BibTeX JSON-RPC endpoint (DOI first, title as the
fallback), and exports the found items as Better BibLaTeX into
shared/references.bib. Works it cannot find are printed at the end so
they can be added to the library; the build does not fail on them.

Needs Zotero running with Better BibTeX. Run again after adding the
missing works — the .bib is regenerated from scratch each time.
"""

import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LITERATURE = ROOT.parent / "docs" / "en" / "dev" / "literature.md"
OUT = ROOT / "shared" / "references.bib"
RPC = "http://localhost:23119/better-bibtex/json-rpc"


def rpc(method, params):
    req = urllib.request.Request(
        RPC,
        data=json.dumps({"jsonrpc": "2.0", "method": method,
                         "params": params, "id": 1}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read())
    if "error" in body:
        raise RuntimeError(f"{method}: {body['error']}")
    return body["result"]


def parse_works():
    """Table rows: | Authors (Year). *Title*. Venue. | [doi](url) |"""
    works = []
    for line in LITERATURE.read_text(encoding="utf-8").splitlines():
        m = re.match(r"\|\s*(.+?)\s*\|\s*\[([^\]]+)\]\(https://doi\.org/",
                     line)
        if not m:
            continue
        desc, doi = m.group(1), m.group(2)
        t = re.search(r"\*(.+?)\*", desc)
        title = re.sub(r"<[^>]+>", "", t.group(1)) if t else None
        works.append({"desc": re.sub(r"[*|]", "", desc)[:90],
                      "doi": doi.lower(), "title": title})
    return works


def norm(s):
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())


def find(work):
    # DOI is the strongest key; quicksearch covers the DOI field.
    for hit in rpc("item.search", [work["doi"]]):
        if (hit.get("DOI") or "").lower().rstrip(".") == work["doi"]:
            return hit
    if work["title"]:
        wanted = norm(work["title"])
        for hit in rpc("item.search", [work["title"]]):
            got_doi = (hit.get("DOI") or "").lower()
            if got_doi == work["doi"] or norm(hit.get("title")) == wanted:
                return hit
    return None


def main():
    works = parse_works()
    found, missing = [], []
    for w in works:
        hit = find(w)
        if hit and hit.get("citation-key"):
            found.append((w, hit["citation-key"]))
        else:
            missing.append(w)

    keys = sorted({k for _, k in found})
    bib = rpc("item.export", [keys, "Better BibLaTeX"])
    # BBT returns either the string or [status, headers, string]
    if isinstance(bib, (list, tuple)):
        bib = bib[-1]
    OUT.write_text(bib, encoding="utf-8")
    print(f"{len(found)} of {len(works)} works found; "
          f"{len(keys)} entries -> {OUT.relative_to(ROOT)}")

    if missing:
        print("\nNOT in the Zotero library (add and re-run):")
        for w in missing:
            print(f"  - {w['desc']}")
            print(f"    doi:{w['doi']}")
    print("\nCitation keys:")
    for w, k in found:
        print(f"  {k:35s} {w['desc'][:70]}")


if __name__ == "__main__":
    sys.exit(main())
