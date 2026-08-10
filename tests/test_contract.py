"""The reader must agree with the contract's own reference expectations.

``expected.json`` states what a correct reader produces for each of the nine
reference files.  Reproducing it means this implementation agrees with the
plugin on encoding, quoting, line breaks inside fields and the byte-order mark
-- the four things that usually differ between two CSV implementations.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import qdapy
from qdapy.read import ContractError

DATA = Path(str(qdapy.example("zotqda-fragments.csv"))).parent
EXPECTED = json.loads((DATA / "expected.json").read_text(encoding="utf-8"))


def test_contract_is_the_one_the_samples_were_made_from():
    ct = qdapy.contract()
    assert ct["contract"] == EXPECTED["contract"]
    assert ct["version"] == EXPECTED["version"]
    assert len(ct["formats"]) == 9


def test_every_format_declares_a_stamp_and_columns():
    for name, spec in qdapy.formats().items():
        assert spec["id"] == f"{name}/{qdapy.contract()['version']}"
        assert spec["stampColumn"] == "zotqdaFormat"
        assert spec["columns"], f"{name} declares no columns"
        assert spec["columns"][0]["key"] == "zotqdaFormat"
        assert spec["grain"]


def test_the_stamp_column_is_the_same_everywhere():
    assert qdapy.stamp_column() == "zotqdaFormat"


def test_example_lists_all_nine_reference_files():
    """The nine contract files, and nothing missing from them.

    `example()` also offers sample.qdpx (REFI route) and the two demo-study
    files (shared with qdaR, exercised by test_demo_export.py). None of the
    three belongs to the contract, so they are named here explicitly instead
    of being swept up by a count.
    """
    names = qdapy.example()
    assert set(EXPECTED["files"]) <= set(names)
    assert len([n for n in names if n.endswith(".csv")]) == 11
    assert set(names) - set(EXPECTED["files"]) == {
        "sample.qdpx",
        "zotqda-fragments-demo.csv",
        "zotqda-history-demo.csv",
    }


def test_example_refuses_a_name_it_does_not_have():
    with pytest.raises(FileNotFoundError, match="no such reference file"):
        qdapy.example("nope.csv")


@pytest.mark.parametrize("name", sorted(EXPECTED["files"]))
def test_reader_reproduces_the_reference_file(name):
    expected = EXPECTED["files"][name]
    df = qdapy.read(qdapy.example(name))

    assert list(df.columns) == expected["columns"]
    assert len(df) == len(expected["rows"])
    kind, version = expected["format"].split("/")
    assert df.attrs["qda_format"] == kind
    assert df.attrs["qda_version"] == int(version)

    numeric = {
        c["key"]
        for c in qdapy.formats()[kind]["columns"]
        if c["type"] == "number"
    }
    for i, want in enumerate(expected["rows"]):
        for column, value in want.items():
            got = df.iloc[i][column]
            if column in numeric:
                # a numeric column may legitimately be empty: `weight` when
                # nobody set one, `positionStart` on a PDF segment
                if value == "":
                    assert pd.isna(got) or got == "", (name, i, column)
                else:
                    assert float(got) == pytest.approx(float(value)), (name, i, column)
            else:
                # the awkward cases live here: embedded quotes, the delimiter
                # and a line break inside a field
                assert got == value, (name, i, column)


def test_a_field_with_a_line_break_survived():
    df = qdapy.read(qdapy.example("zotqda-fragments.csv"))
    assert any("\n" in str(v) for v in df["citekey"])


def test_a_field_with_the_delimiter_and_quotes_survived():
    df = qdapy.read(qdapy.example("zotqda-fragments.csv"))
    joined = " ".join(str(v) for v in df["citekey"])
    assert ";" in joined
    assert '"' in joined


def test_contract_can_be_read_from_an_explicit_path():
    ct = qdapy.contract(DATA / "exchange-v1.json")
    assert ct["version"] == qdapy.contract()["version"]


def test_a_missing_contract_file_is_an_error():
    with pytest.raises(FileNotFoundError, match="exchange contract not found"):
        qdapy.contract("/nonexistent/exchange-v1.json")


def test_the_position_columns_arrive_in_both_shapes():
    """E37.1 - unitizing reliability is impossible without these."""
    df = qdapy.read_fragments(qdapy.example("zotqda-fragments.csv"))
    for column in ("positionKind", "positionStart", "positionEnd",
                   "positionPage", "positionRects"):
        assert column in df.columns
    assert list(df["positionKind"]) == ["text", "pdf"]
    text_row = df[df["positionKind"] == "text"].iloc[0]
    assert float(text_row["positionStart"]) == 120
    assert float(text_row["positionEnd"]) == 180
    pdf_row = df[df["positionKind"] == "pdf"].iloc[0]
    assert float(pdf_row["positionPage"]) == 3
    assert pdf_row["positionRects"] == "10.5 20 110 32|10.5 33 90 45"


def test_an_export_without_the_position_columns_is_rejected(tmp_path):
    """They are part of the contract, so a file lacking them is broken.

    Nothing may rely on a fallback path here: unitizing reliability is
    impossible without a position, and a reader that silently accepted the
    file would produce agreement figures with the segmentation question
    quietly dropped.
    """
    ct = qdapy.contract()
    cols = [c["key"] for c in ct["formats"]["fragments"]["columns"]
            if not c["key"].startswith("position")]
    path = tmp_path / "old.csv"
    path.write_text("\ufeff" + ",".join(cols) + "\n"
                    + ",".join("fragments/1" if c == "zotqdaFormat" else "x"
                               for c in cols) + "\n", encoding="utf-8")
    with pytest.raises(ContractError, match="missing contract columns"):
        qdapy.read_fragments(path)
    # and the caller can still look at it if they know what they are doing
    df = qdapy.read(path, "fragments", strict=False)
    assert "positionKind" not in df.columns


def test_no_column_is_optional():
    for kind in qdapy.formats():
        assert not any(c.get("optional")
                       for c in qdapy.formats()[kind]["columns"]), kind
