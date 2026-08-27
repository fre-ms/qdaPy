"""E36.5: the COREQ checklist, and what it can and cannot fill in."""

from __future__ import annotations

import pandas as pd

import qdapy
from qdapy.reporting import ITEMS


def test_all_thirty_two_items_are_present_and_numbered_once():
    assert len(ITEMS) == 32
    numbers = [int(i[0]) for i in ITEMS]
    assert numbers == list(range(1, 33))


def test_the_three_domains_have_the_published_sizes():
    counts = {}
    for _, domain, *_ in ITEMS:
        counts[domain] = counts.get(domain, 0) + 1
    assert counts == {"Research team and reflexivity": 8,
                      "Study design": 15,
                      "Analysis and findings": 9}


def test_an_item_is_reproduced_verbatim():
    # item 22 is the one this package can answer, so it had better be the
    # question the reviewer is looking at
    item22 = next(i for i in ITEMS if i[0] == "22")
    assert item22[3] == "Data saturation"
    assert item22[4] == "Was data saturation discussed?"


def test_the_checklist_fills_what_the_data_knows_and_no_more():
    frag = qdapy.read_fragments(qdapy.example("easyqda-fragments.csv"))
    cq = qdapy.coreq(frag)
    assert len(cq) == 32
    filled = set(cq[cq["filled"]]["item"])
    # documents, coders, software, quotations -- but not saturation or the
    # coding tree, which need the other two exports
    assert filled == {12, 24, 27, 29}
    assert (cq[~cq["filled"]]["answer"] == "").all()


def test_the_saturation_item_arrives_with_a_history():
    h = pd.DataFrame({
        "ts": [f"2026-01-{d:02d}T09:00:00Z" for d in range(1, 13)],
        "user": "ann", "action": "add",
        "code": ["A", "B", "C", "D", "E", "F", "G", "H", "A", "B", "A", "B"],
        "citekey": [f"d{i}" for i in range(1, 13)],
    })
    cq = qdapy.coreq(history=h)
    row = cq[cq["item"] == 22].iloc[0]
    assert row["filled"]
    assert "saturation" in row["answer"]


def test_the_coding_tree_item_arrives_with_a_codebook():
    cb = qdapy.read_codebook(qdapy.example("easyqda-codebook.csv"))
    cq = qdapy.coreq(codebook=cb)
    row = cq[cq["item"] == 25].iloc[0]
    assert row["filled"]
    assert "codes over" in row["answer"]


def test_the_software_item_names_both_tools_and_can_be_overridden():
    cq = qdapy.coreq()
    row = cq[cq["item"] == 27].iloc[0]
    assert "zotQDA" in row["answer"] and "qdaPy" in row["answer"]
    custom = qdapy.coreq(software="MAXQDA 24")
    assert custom[custom["item"] == 27].iloc[0]["answer"] == "MAXQDA 24"


def test_the_markdown_carries_every_item_and_the_citation():
    frag = qdapy.read_fragments(qdapy.example("easyqda-fragments.csv"))
    md = qdapy.coreq_markdown(qdapy.coreq(frag))
    assert "doi:10.1093/intqhc/mzm042" in md
    for number, _, _, name, _ in ITEMS:
        assert f"**{number}. {name}**" in md
    assert md.count("*To be completed.*") == 28


# ---- SRQR -----------------------------------------------------------


def test_all_twenty_one_standards_are_present_and_numbered_once():
    from qdapy.reporting import SRQR_ITEMS
    assert len(SRQR_ITEMS) == 21
    assert [i[0] for i in SRQR_ITEMS] == [f"S{n}" for n in range(1, 22)]


def test_the_srqr_sections_have_the_published_sizes():
    from qdapy.reporting import SRQR_ITEMS
    counts = {}
    for _, section, *_ in SRQR_ITEMS:
        counts[section] = counts.get(section, 0) + 1
    assert counts == {"Title and abstract": 2, "Introduction": 2,
                      "Methods": 11, "Results/findings": 2,
                      "Discussion": 2, "Other": 2}


def test_an_srqr_standard_is_reproduced_verbatim():
    from qdapy.reporting import SRQR_ITEMS
    s15 = next(i for i in SRQR_ITEMS if i[0] == "S15")
    assert s15[2] == "Techniques to enhance trustworthiness"
    assert "audit trail" in s15[3]


def test_srqr_fills_what_the_data_knows():
    frag = qdapy.read_fragments(qdapy.example("easyqda-fragments.csv"))
    sq = qdapy.srqr(frag)
    assert len(sq) == 21
    assert set(sq[sq["filled"]]["item"]) == {"S12", "S13", "S14", "S17"}


def test_srqr_has_a_home_for_the_agreement_figure_where_coreq_has_none():
    # the difference that decides which checklist to use when both are allowed
    h = pd.DataFrame({
        "ts": [f"2026-01-{d:02d}T09:00:00Z" for d in range(1, 6)],
        "user": "ann", "action": "add", "code": list("ABCDE"),
        "citekey": [f"d{i}" for i in range(1, 6)],
    })
    sq = qdapy.srqr(history=h)
    s15 = sq[sq["item"] == "S15"].iloc[0]
    assert s15["filled"]
    assert "audit trail" in s15["answer"]
    assert "agreement" in s15["answer"]


def test_the_saturation_standard_warns_which_saturation_it_means():
    h = pd.DataFrame({
        "ts": [f"2026-01-{d:02d}T09:00:00Z" for d in range(1, 13)],
        "user": "ann", "action": "add",
        "code": ["A", "B", "C", "D", "E", "F", "G", "H", "A", "B", "A", "B"],
        "citekey": [f"d{i}" for i in range(1, 13)],
    })
    answer = qdapy.srqr(history=h)
    answer = answer[answer["item"] == "S8"].iloc[0]["answer"]
    assert "code saturation" in answer
    assert "not meaning saturation" in answer


def test_the_srqr_markdown_carries_every_standard_and_the_citation():
    frag = qdapy.read_fragments(qdapy.example("easyqda-fragments.csv"))
    md = qdapy.srqr_markdown(qdapy.srqr(frag))
    assert "doi:10.1097/ACM.0000000000000388" in md
    from qdapy.reporting import SRQR_ITEMS
    for number, _, name, _ in SRQR_ITEMS:
        assert f"**{number} {name}**" in md
