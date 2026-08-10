"""Reading is where a wrong answer is cheapest to prevent."""

from __future__ import annotations

import pandas as pd
import pytest

import qdapy
from qdapy.read import ContractError, _sniff_delimiter

FRAG_COLS = [c["key"] for c in qdapy.formats()["fragments"]["columns"]]


def write(tmp_path, name, text, *, bom=True):
    path = tmp_path / name
    path.write_text(("﻿" if bom else "") + text, encoding="utf-8")
    return path


def fragments_csv(rows, *, sep=",", stamp="fragments/1", columns=None):
    cols = columns if columns is not None else FRAG_COLS
    lines = [sep.join(cols)]
    for row in rows:
        lines.append(sep.join(str(row.get(c, "")) for c in cols))
    return "\n".join(lines) + "\n"


def test_reads_a_semicolon_file(tmp_path):
    text = fragments_csv([{"zotqdaFormat": "fragments/1", "code": "A",
                           "weight": 2}], sep=";")
    df = qdapy.read(write(tmp_path, "semi.csv", text))
    assert list(df.columns) == FRAG_COLS
    assert df.iloc[0]["code"] == "A"
    assert df.iloc[0]["weight"] == 2


def test_reads_a_file_without_a_byte_order_mark(tmp_path):
    text = fragments_csv([{"zotqdaFormat": "fragments/1", "code": "A"}])
    df = qdapy.read(write(tmp_path, "nobom.csv", text, bom=False))
    assert df.columns[0] == "zotqdaFormat"


def test_the_mark_never_ends_up_in_a_column_name(tmp_path):
    text = fragments_csv([{"zotqdaFormat": "fragments/1", "code": "A"}])
    df = qdapy.read(write(tmp_path, "bom.csv", text))
    assert not any(c.startswith("﻿") for c in df.columns)


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("a,b,c", ","),
        ("a;b;c", ";"),
        ('a;b;"c,d"', ";"),          # a comma inside a quoted field
        ('a,b,"c;d"', ","),          # and the other way round
    ],
)
def test_delimiter_detection(header, expected):
    assert _sniff_delimiter(header) == expected


def test_numeric_contract_columns_become_numbers(tmp_path):
    text = fragments_csv([{"zotqdaFormat": "fragments/1", "code": "A",
                           "weight": 3}])
    df = qdapy.read(write(tmp_path, "num.csv", text))
    # integral or floating, but not text -- pandas narrows to int when it can
    assert df["weight"].dtype.kind in "if"
    assert df.iloc[0]["weight"] == 3


def test_an_unparsable_number_becomes_missing_not_zero(tmp_path):
    text = fragments_csv([{"zotqdaFormat": "fragments/1", "code": "A",
                           "weight": "viel"}])
    df = qdapy.read(write(tmp_path, "bad.csv", text))
    assert pd.isna(df.iloc[0]["weight"])


def test_a_file_without_the_stamp_column_is_refused(tmp_path):
    path = write(tmp_path, "foreign.csv", "code,text\nA,hallo\n")
    with pytest.raises(ContractError, match="not a zotQDA exchange file"):
        qdapy.read(path)


def test_an_unknown_kind_is_refused(tmp_path):
    text = fragments_csv([{"zotqdaFormat": "sentiments/1", "code": "A"}])
    with pytest.raises(ContractError, match="unknown export kind"):
        qdapy.read(write(tmp_path, "unknown.csv", text))


def test_a_newer_version_is_refused_rather_than_guessed_at(tmp_path):
    text = fragments_csv([{"zotqdaFormat": "fragments/2", "code": "A"}])
    with pytest.raises(ContractError, match="please update qdaPy"):
        qdapy.read(write(tmp_path, "future.csv", text))


def test_an_older_version_is_still_read(tmp_path):
    text = fragments_csv([{"zotqdaFormat": "fragments/0", "code": "A"}])
    df = qdapy.read(write(tmp_path, "old.csv", text))
    assert df.attrs["qda_version"] == 0


def test_a_file_mixing_two_formats_is_refused(tmp_path):
    text = fragments_csv([{"zotqdaFormat": "fragments/1", "code": "A"},
                          {"zotqdaFormat": "codebook/1", "code": "B"}])
    with pytest.raises(ContractError, match="mixes formats"):
        qdapy.read(write(tmp_path, "mixed.csv", text))


def test_the_wrong_export_is_caught_when_a_format_is_demanded():
    with pytest.raises(ContractError, match="expected a 'fragments' export"):
        qdapy.read_fragments(qdapy.example("zotqda-codebook.csv"))


def test_missing_contract_columns_are_an_error_but_can_be_waived(tmp_path):
    columns = ["zotqdaFormat", "code", "annotationKey", "codedBy"]
    text = fragments_csv([{"zotqdaFormat": "fragments/1", "code": "A"}],
                         columns=columns)
    path = write(tmp_path, "short.csv", text)
    with pytest.raises(ContractError, match="missing contract columns"):
        qdapy.read(path)
    df = qdapy.read(path, strict=False)
    assert list(df.columns) == columns


def test_extra_columns_are_accepted_without_complaint(tmp_path):
    columns = [*FRAG_COLS, "somethingNew"]
    text = fragments_csv([{"zotqdaFormat": "fragments/1", "code": "A",
                           "somethingNew": "x"}], columns=columns)
    df = qdapy.read(write(tmp_path, "extra.csv", text))
    assert df.iloc[0]["somethingNew"] == "x"


def test_an_empty_file_is_an_error(tmp_path):
    with pytest.raises(ContractError, match="empty file"):
        qdapy.read(write(tmp_path, "empty.csv", ""))


def test_a_header_without_rows_is_an_error(tmp_path):
    text = fragments_csv([])
    with pytest.raises(ContractError, match="no rows"):
        qdapy.read(write(tmp_path, "headeronly.csv", text))


@pytest.mark.parametrize(
    ("reader", "name"),
    [
        (qdapy.read_fragments, "zotqda-fragments.csv"),
        (qdapy.read_uncoded, "zotqda-uncoded.csv"),
        (qdapy.read_codebook, "zotqda-codebook.csv"),
        (qdapy.read_history, "zotqda-history.csv"),
        (qdapy.read_mapping, "zotqda-konsens-abbildung.csv"),
    ],
)
def test_each_named_reader_accepts_its_own_file(reader, name):
    df = reader(qdapy.example(name))
    assert not df.empty


def test_the_mapping_adds_a_column_and_leaves_the_coding_alone():
    frag = pd.DataFrame({
        "annotationKey": ["s1", "s2"],
        "codedBy": ["ann", "bob"],
        "code": ["Belastung", "Stress"],
    })
    mapping = pd.DataFrame({
        "coder": ["ann", "bob"],
        "coderCode": ["Belastung", "Stress"],
        "consensusCode": ["Belastung", "Belastung"],
    })
    out = qdapy.apply_mapping(frag, mapping)
    assert list(out["code"]) == ["Belastung", "Stress"]        # untouched
    assert list(out["consensusCode"]) == ["Belastung", "Belastung"]


def test_an_unmapped_coding_gets_an_empty_consensus_code_not_a_wrong_one():
    frag = pd.DataFrame({
        "annotationKey": ["s1"], "codedBy": ["cat"], "code": ["Neu"],
    })
    mapping = pd.DataFrame({
        "coder": ["ann"], "coderCode": ["Belastung"],
        "consensusCode": ["Belastung"],
    })
    out = qdapy.apply_mapping(frag, mapping)
    assert out.iloc[0]["consensusCode"] == ""


def test_the_mapping_is_keyed_on_the_coder_too():
    # the same code name from two coders can mean two different things
    frag = pd.DataFrame({
        "annotationKey": ["s1", "s2"],
        "codedBy": ["ann", "bob"],
        "code": ["Druck", "Druck"],
    })
    mapping = pd.DataFrame({
        "coder": ["ann", "bob"],
        "coderCode": ["Druck", "Druck"],
        "consensusCode": ["Belastung", "Zeitnot"],
    })
    out = qdapy.apply_mapping(frag, mapping)
    assert list(out["consensusCode"]) == ["Belastung", "Zeitnot"]


def test_a_mapping_without_its_contract_columns_is_an_error():
    frag = pd.DataFrame({"codedBy": ["ann"], "code": ["A"]})
    with pytest.raises(KeyError, match="coderCode"):
        qdapy.apply_mapping(frag, pd.DataFrame({"coder": ["ann"]}))
