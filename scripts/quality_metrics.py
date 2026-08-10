#!/usr/bin/env python3
"""Reproducible code-quality metric snapshot for qdaPy.

Runs the pinned commands the quality review used and writes a
machine-readable JSON snapshot plus a human summary. Diff two snapshots to
see whether a change moved a metric against the project's OWN history -
Nagappan, Ball and Zeller (2006) doi:10.1145/1134285.1134349 found no set of
metrics that fits all projects and recommend calibrating against your own
baseline instead of a universal threshold.

WHAT IT MEASURES (each tool independent; a missing tool is recorded, not fatal)
  * radon       mean cyclomatic complexity, grade histogram, functions over
                CC 10, module maintainability index
  * complexipy  max cognitive complexity and how many functions exceed 15
  * ruff        finding counts per rule, under the project's own config
  * mypy        error count under the project's own config (expected: zero)
  * interrogate docstring coverage, plus how many PUBLIC top-level
                definitions lack one - the number that actually matters
  * vulture     unreferenced code candidates
  * coverage    line coverage total and per module

USAGE
  python scripts/quality_metrics.py                 # human summary
  python scripts/quality_metrics.py --json out.json # + write the snapshot
  python scripts/quality_metrics.py --json -        # JSON only, no summary
  python scripts/quality_metrics.py --baseline b.json   # compare and exit 1
                                                        # on a regression

DELIBERATELY NOT A PASS/FAIL ORACLE for the complexity numbers. Radjenovic et
al. (2013) doi:10.1016/j.infsof.2013.02.009 and Scalabrino et al. (2021)
doi:10.1109/TSE.2019.2901468 both find that no static metric captures
understandability reliably. The hard gates live in the tools' own configs
(ruff C901, mypy); what is here is a trend instrument.

Read `MI` with care: radon reports an SEI-normalised 0-100 variant whose own
bands are A above 19. Coleman et al. (1994) doi:10.1109/2.303623 give the
"below 65 is difficult to maintain" rule of thumb for the UNNORMALISED
formula; the two scales must not be compared.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "qdapy"

# Metrics that must not get worse. Everything else is recorded for the trend
# but does not fail a comparison: a rising function count is not a defect.
REGRESSION_KEYS = (
    ("ruff", "total_findings"),
    ("mypy", "errors"),
    ("radon", "functions_over_cc10"),
    ("complexipy", "functions_over_15"),
    ("docstrings", "public_missing"),
)


def _run(cmd: list[str]) -> tuple[int, str]:
    """Run a command, returning (returncode, stdout+stderr).

    Never raises: a missing tool is data about the environment, not a crash.
    """
    try:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except FileNotFoundError as exc:
        return 127, str(exc)


def _py(*args: str) -> tuple[int, str]:
    return _run([sys.executable, "-m", *args])


def radon_metrics() -> dict[str, Any]:
    """Cyclomatic complexity per block, plus the maintainability index."""
    rc, out = _py("radon", "cc", "-s", "-j", str(SRC))
    if rc != 0 or not out.strip():
        return {"error": "radon cc failed", "detail": out[:400]}
    data = json.loads(out)
    grades: dict[str, int] = {}
    worst: list[dict[str, Any]] = []
    total = n = 0
    for path, blocks in data.items():
        if not isinstance(blocks, list):
            continue
        for b in blocks:
            grades[b["rank"]] = grades.get(b["rank"], 0) + 1
            total += b["complexity"]
            n += 1
            if b["complexity"] > 10:
                worst.append({"file": Path(path).name, "name": b["name"],
                              "line": b["lineno"], "cc": b["complexity"]})
    worst.sort(key=lambda r: -r["cc"])

    mi: dict[str, float] = {}
    rc_mi, out_mi = _py("radon", "mi", "-j", str(SRC))
    if rc_mi == 0 and out_mi.strip():
        for path, rec in json.loads(out_mi).items():
            if isinstance(rec, dict) and "mi" in rec:
                mi[Path(path).name] = round(rec["mi"], 2)
    return {
        "blocks": n,
        "mean_cc": round(total / n, 2) if n else None,
        "grade_histogram": dict(sorted(grades.items())),
        "functions_over_cc10": len(worst),
        "over_cc10": worst,
        "mi_by_module": dict(sorted(mi.items(), key=lambda kv: kv[1])),
        "mi_min": min(mi.values()) if mi else None,
    }


def complexipy_metrics() -> dict[str, Any]:
    """Cognitive complexity: nesting-sensitive, unlike cyclomatic."""
    rc, out = _run(["complexipy", str(SRC), "--output-csv", "-q"])
    csv_path = ROOT / "complexipy.csv"
    if not csv_path.exists():
        return {"error": "complexipy produced no CSV", "detail": out[:400],
                "returncode": rc}
    rows = []
    try:
        import csv as _csv

        with csv_path.open(encoding="utf-8") as fh:
            for r in _csv.DictReader(fh):
                rows.append((int(r["Cognitive Complexity"]),
                             r["File Name"], r["Function Name"]))
    finally:
        csv_path.unlink(missing_ok=True)
    rows.sort(reverse=True)
    return {
        "functions": len(rows),
        "max_cognitive": rows[0][0] if rows else None,
        # 15 is the SonarQube default, a tool convention and not a finding
        # from the literature; recorded as a trend line, not a gate.
        "functions_over_15": sum(1 for c, _, _ in rows if c > 15),
        "worst": [{"file": f, "name": n, "cognitive": c}
                  for c, f, n in rows[:5]],
    }


def ruff_metrics() -> dict[str, Any]:
    """Findings under the project's own ruff config (includes C901)."""
    rc, out = _py("ruff", "check", "src", "tests", "--statistics")
    if rc == 127:
        return {"error": out}
    counts: dict[str, int] = {}
    for line in out.splitlines():
        m = re.match(r"\s*(\d+)\s+([A-Z]+\d+)\s", line)
        if m:
            counts[m.group(2)] = int(m.group(1))
    return {"findings_by_rule": dict(sorted(counts.items())),
            "total_findings": sum(counts.values())}


def mypy_metrics() -> dict[str, Any]:
    """Type errors under the project's own config. Expected: zero."""
    rc, out = _py("mypy")
    if rc == 127:
        return {"error": out}
    m = re.search(r"Found (\d+) error", out)
    return {"errors": int(m.group(1)) if m else 0,
            "clean": "Success" in out}


def docstring_metrics() -> dict[str, Any]:
    """Coverage overall, and the number that matters: PUBLIC definitions.

    interrogate counts private helpers and closures, so its percentage
    understates a package whose whole public surface is documented. Both
    numbers are recorded; only the public one is a regression key.
    """
    rc, out = _run(["interrogate", str(SRC)])
    pct = None
    m = re.search(r"actual:\s*([\d.]+)%", out) or re.search(r"([\d.]+)%", out)
    if m:
        pct = float(m.group(1))

    missing: list[str] = []
    for p in sorted(SRC.glob("*.py")):
        for node in ast.parse(p.read_text(encoding="utf-8")).body:
            if not isinstance(node, ast.FunctionDef | ast.ClassDef):
                continue
            if node.name.startswith("_") or ast.get_docstring(node):
                continue
            missing.append(f"{p.name}:{node.lineno} {node.name}")
    return {"coverage_pct": pct, "public_missing": len(missing),
            "public_missing_names": missing,
            "note": None if rc != 127 else out}


def vulture_metrics() -> dict[str, Any]:
    """Unreferenced-code candidates.

    Candidates, not findings: vulture cannot see pytest fixtures, dataclass
    fields read by callers, or PEP 562 module hooks. Every hit needs a look.
    """
    rc, out = _run(["vulture", str(SRC), str(ROOT / "tests"),
                    "--min-confidence", "60"])
    if rc == 127:
        return {"error": out}
    hits = [ln.strip() for ln in out.splitlines() if ln.strip()]
    fields = _typeddict_fields()
    real = [h for h in hits
            if not (m := re.search(r"unused variable '(\w+)'", h))
            or m.group(1) not in fields]
    return {"candidates": len(real), "detail": real,
            "typeddict_fields_ignored": len(hits) - len(real)}


def _typeddict_fields() -> set[str]:
    """Field names declared in a TypedDict, which vulture reads as unused."""
    names: set[str] = set()
    for p in SRC.glob("*.py"):
        for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.ClassDef):
                continue
            if not any(getattr(b, "id", getattr(b, "attr", "")) == "TypedDict"
                       for b in node.bases):
                continue
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(
                        stmt.target, ast.Name):
                    names.add(stmt.target.id)
    return names


def coverage_metrics() -> dict[str, Any]:
    """Line coverage, run fresh so the snapshot is self-contained."""
    rc, out = _py("coverage", "run", "--source", str(SRC),
                  "-m", "pytest", "-p", "no:cacheprovider")
    if rc == 127:
        return {"error": out}
    tests = re.search(r"(\d+) passed", out)
    rc_j, out_j = _py("coverage", "json", "-o", "-", "--quiet")
    if rc_j != 0 or not out_j.strip():
        return {"error": "coverage json failed", "detail": out_j[:400]}
    data = json.loads(out_j)
    per_module = {
        Path(f).name: round(rec["summary"]["percent_covered"], 1)
        for f, rec in sorted(data.get("files", {}).items())
    }
    return {
        "total_pct": round(data["totals"]["percent_covered"], 1),
        "tests_passed": int(tests.group(1)) if tests else None,
        "suite_green": rc == 0,
        "by_module_pct": dict(sorted(per_module.items(),
                                     key=lambda kv: kv[1])),
    }


def collect() -> dict[str, Any]:
    return {
        "radon": radon_metrics(),
        "complexipy": complexipy_metrics(),
        "ruff": ruff_metrics(),
        "mypy": mypy_metrics(),
        "docstrings": docstring_metrics(),
        "vulture": vulture_metrics(),
        "coverage": coverage_metrics(),
    }


def _summary(snap: dict[str, Any]) -> str:
    r, cx, rf = snap["radon"], snap["complexipy"], snap["ruff"]
    my, ds, vu, cov = (snap["mypy"], snap["docstrings"], snap["vulture"],
                       snap["coverage"])
    out = ["qdaPy quality snapshot", "-" * 52]
    if "mean_cc" in r:
        out.append(f"radon       mean CC {r['mean_cc']} over {r['blocks']} blocks"
                   f"  |  {r['functions_over_cc10']} > CC10"
                   f"  |  grades {r['grade_histogram']}")
        out.append(f"            MI min {r['mi_min']} (radon scale, A above 19)")
    if "max_cognitive" in cx:
        out.append(f"complexipy  max CogC {cx['max_cognitive']}"
                   f"  |  {cx['functions_over_15']} > 15")
    if "total_findings" in rf:
        by_rule = rf["findings_by_rule"] or ""
        out.append(f"ruff        {rf['total_findings']} findings {by_rule}")
    if "errors" in my:
        out.append(f"mypy        {my['errors']} errors"
                   f"{'  (clean)' if my.get('clean') else ''}")
    if ds.get("coverage_pct") is not None:
        out.append(f"docstrings  {ds['coverage_pct']}% overall"
                   f"  |  {ds['public_missing']} public definitions undocumented")
    if "candidates" in vu:
        skipped = vu.get("typeddict_fields_ignored", 0)
        out.append(f"vulture     {vu['candidates']} candidates"
                   f"  ({skipped} TypedDict fields ignored)")
    if "total_pct" in cov:
        out.append(f"coverage    {cov['total_pct']}%"
                   f"  |  {cov['tests_passed']} tests"
                   f"  |  suite {'green' if cov['suite_green'] else 'RED'}")
    return "\n".join(out)


def _compare(snap: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    """Regressions against a baseline, on the keys where higher is worse."""
    bad = []
    for section, key in REGRESSION_KEYS:
        now = snap.get(section, {}).get(key)
        was = baseline.get(section, {}).get(key)
        if isinstance(now, int) and isinstance(was, int) and now > was:
            bad.append(f"{section}.{key}: {was} -> {now}")
    cov_now = snap.get("coverage", {}).get("total_pct")
    cov_was = baseline.get("coverage", {}).get("total_pct")
    if (isinstance(cov_now, int | float) and isinstance(cov_was, int | float)
            and cov_now < cov_was - 0.5):   # half a point of measurement noise
        bad.append(f"coverage.total_pct: {cov_was} -> {cov_now}")
    if snap.get("coverage", {}).get("suite_green") is False:
        bad.append("the test suite is not green")
    return bad


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", metavar="PATH",
                    help="write the snapshot to PATH ('-' for stdout)")
    ap.add_argument("--baseline", metavar="PATH",
                    help="compare against an earlier snapshot; exit 1 on a "
                         "regression in the gated keys")
    args = ap.parse_args(argv)

    snap = collect()
    if args.json == "-":
        print(json.dumps(snap, indent=2))
    else:
        if args.json:
            Path(args.json).write_text(json.dumps(snap, indent=2) + "\n",
                                       encoding="utf-8")
            print(f"-> {args.json}")
        print(_summary(snap))

    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        bad = _compare(snap, baseline)
        if bad:
            print("\nREGRESSION against " + args.baseline, file=sys.stderr)
            for line in bad:
                print("  " + line, file=sys.stderr)
            return 1
        print(f"\nno regression against {args.baseline}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
