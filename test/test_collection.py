"""The collection reader must agree with the contract's reference tables.

The easyQDA-CSV-Collection (``easyqda-collection`` contract) ships one sample
CSV per table plus ``expected.json``; reproducing it proves this reader agrees
with the plugin on the stamp, the columns, quoting and the byte-order mark.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

import qdapy
from qdapy.read import ContractError

DATA = Path(str(qdapy.example("collection-v1.json"))).parent
SAMPLES = DATA / "collection-samples"


def test_contract_is_the_collection_contract():
    ct = qdapy.collection_contract()
    assert ct["contract"] == "easyqda-collection"
    assert ct["version"] == 1
    for core in ("project", "codes", "selections", "codings", "history"):
        assert core in ct["tables"]


def test_every_sample_table_reads_and_matches_expected():
    expected = json.loads((SAMPLES / "expected.json").read_text(encoding="utf-8"))
    ct = qdapy.collection_contract()
    for file, exp in expected["files"].items():
        table = exp["table"]
        df = qdapy.read_collection_table(SAMPLES / file, table)
        # the stamp column plus every contract column is present, in order
        assert list(df.columns) == exp["columns"]
        assert df.attrs["qda_table"] == table
        assert df.attrs["qda_version"] == 1
        # the stamp value is the table-scoped id
        assert (df[ct["tables"][table]["stampColumn"]] == exp["format"]).all()


def test_read_whole_collection_from_a_directory(tmp_path):
    # assemble a collection layout (tables/<t>.csv) from the samples
    (tmp_path / "tables").mkdir()
    for csv in SAMPLES.glob("*.csv"):
        (tmp_path / "tables" / csv.name).write_bytes(csv.read_bytes())
    (tmp_path / "datapackage.json").write_bytes((SAMPLES / "datapackage.json").read_bytes())

    tables = qdapy.read_collection(tmp_path)
    for core in ("codes", "selections", "codings", "history"):
        assert core in tables
        assert not tables[core].empty


def test_read_whole_collection_from_a_zip(tmp_path):
    zpath = tmp_path / "demo.easyqda-csv.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for csv in SAMPLES.glob("*.csv"):
            zf.writestr("tables/" + csv.name, csv.read_bytes())
        zf.writestr("datapackage.json", (SAMPLES / "datapackage.json").read_bytes())

    tables = qdapy.read_collection(zpath)
    assert "codes" in tables and "codings" in tables


def test_a_newer_version_is_refused(tmp_path):
    p = tmp_path / "codes.csv"
    p.write_text("easyqdaFormat,codeId,parentId,name,path,color,isCodable,abbrev,memo\r\n"
                 "collection-codes/2,X,,A,A,,1,,\r\n", encoding="utf-8")
    with pytest.raises(ContractError, match="please update qdaPy"):
        qdapy.read_collection_table(p, "codes")


def test_wrong_stamp_is_rejected(tmp_path):
    p = tmp_path / "codes.csv"
    p.write_text("easyqdaFormat,codeId,parentId,name,path,color,isCodable,abbrev,memo\r\n"
                 "collection-selections/1,X,,A,A,,1,,\r\n", encoding="utf-8")
    with pytest.raises(ContractError, match="expected"):
        qdapy.read_collection_table(p, "codes")
