#!/usr/bin/env python3
"""Build the REFI-QDA fixture both packages read.

The archive is written with fixed timestamps and no compression variance, so
the bytes are reproducible: qdaR ships the same file and a test asserts the
two are byte-identical, the same arrangement the coefficient references use.

The project deliberately contains what a naive reader gets wrong:

* a source whose text is inline (`PlainTextContent`) and one that points at an
  internal file (`plainTextPath`), because tools differ on which they write;
* a selection carrying two codings by one coder, so the multiple-coding path
  is exercised;
* a selection with no coding at all, which is the only way a .qdpx can express
  an uncoded segment;
* two coders on the same two selections, so agreement can be computed;
* a PDF source with a rectangle, which has no continuum to measure on;
* a picture source, a case and a note, none of which become rows - they must
  be counted and reported, not dropped in silence.

Run it from the package root; it writes into both packages when qdaR is
beside this one.
"""

from __future__ import annotations

import json
import pathlib
import sys
import zipfile

HERE = pathlib.Path(__file__).resolve().parents[1]

TEXT_A = (
    "Die Arbeit war fordernd, aber das Team hat getragen. "
    "Nach der Umstellung im Frühjahr wurde es deutlich enger. "
    "Zu Hause konnte ich kaum noch abschalten.\n"
)
TEXT_B = (
    "Am Anfang habe ich die Doppelbelastung unterschätzt. "
    "Die Kinder haben das gespürt, glaube ich.\n"
)

QDE = """<?xml version="1.0" encoding="utf-8"?>
<Project xmlns="urn:QDA-XML:project:1.0"
         name="Belastung im Klinikalltag"
         creatingUserGUID="11111111-1111-1111-1111-111111111111"
         creationDateTime="2026-03-01T09:00:00Z">
  <Users>
    <User guid="11111111-1111-1111-1111-111111111111" name="Ann"/>
    <User guid="22222222-2222-2222-2222-222222222222" name="Bob"/>
  </Users>
  <CodeBook>
    <Codes>
      <Code guid="AAAAAAA1-0000-0000-0000-000000000001" name="Belastung"
            isCodable="true" color="#CC3311">
        <Description>Alles, was als Last geschildert wird.</Description>
        <Code guid="AAAAAAA1-0000-0000-0000-000000000002" name="beruflich"
              isCodable="true">
          <Description>Last, die aus der Arbeit kommt.</Description>
        </Code>
        <Code guid="AAAAAAA1-0000-0000-0000-000000000003" name="privat"
              isCodable="true" color="#EE7733"/>
      </Code>
      <Code guid="AAAAAAA1-0000-0000-0000-000000000004" name="Ressourcen"
            isCodable="true" color="#0077BB"/>
    </Codes>
  </CodeBook>
  <Sources>
    <TextSource guid="BBBBBBB1-0000-0000-0000-000000000001"
                name="Interview 01" plainTextPath="internal://text-01.txt"
                creatingUser="11111111-1111-1111-1111-111111111111"
                creationDateTime="2026-03-01T09:05:00Z">
      <PlainTextSelection guid="CCCCCCC1-0000-0000-0000-000000000001"
                          name="s1" startPosition="0" endPosition="46"
                          creatingUser="11111111-1111-1111-1111-111111111111"
                          creationDateTime="2026-03-02T10:00:00Z">
        <Description>Traegt das Team wirklich?</Description>
        <Coding guid="DDDDDDD1-0000-0000-0000-000000000001"
                creatingUser="11111111-1111-1111-1111-111111111111"
                creationDateTime="2026-03-02T10:00:00Z">
          <CodeRef targetGUID="AAAAAAA1-0000-0000-0000-000000000004"/>
        </Coding>
        <Coding guid="DDDDDDD1-0000-0000-0000-000000000002"
                creatingUser="22222222-2222-2222-2222-222222222222"
                creationDateTime="2026-03-03T11:30:00Z">
          <CodeRef targetGUID="AAAAAAA1-0000-0000-0000-000000000004"/>
        </Coding>
      </PlainTextSelection>
      <PlainTextSelection guid="CCCCCCC1-0000-0000-0000-000000000002"
                          name="s2" startPosition="47" endPosition="106"
                          creatingUser="11111111-1111-1111-1111-111111111111"
                          creationDateTime="2026-03-02T10:05:00Z">
        <Coding guid="DDDDDDD1-0000-0000-0000-000000000003"
                creatingUser="11111111-1111-1111-1111-111111111111"
                creationDateTime="2026-03-02T10:05:00Z">
          <CodeRef targetGUID="AAAAAAA1-0000-0000-0000-000000000002"/>
        </Coding>
        <Coding guid="DDDDDDD1-0000-0000-0000-000000000004"
                creatingUser="22222222-2222-2222-2222-222222222222"
                creationDateTime="2026-03-03T11:35:00Z">
          <CodeRef targetGUID="AAAAAAA1-0000-0000-0000-000000000003"/>
        </Coding>
      </PlainTextSelection>
      <PlainTextSelection guid="CCCCCCC1-0000-0000-0000-000000000003"
                          name="s3" startPosition="107" endPosition="150"
                          creatingUser="11111111-1111-1111-1111-111111111111"
                          creationDateTime="2026-03-02T10:10:00Z"/>
    </TextSource>
    <TextSource guid="BBBBBBB1-0000-0000-0000-000000000002"
                name="Interview 02"
                creatingUser="11111111-1111-1111-1111-111111111111"
                creationDateTime="2026-03-01T09:06:00Z">
      <PlainTextContent>{text_b}</PlainTextContent>
      <PlainTextSelection guid="CCCCCCC1-0000-0000-0000-000000000004"
                          name="s4" startPosition="0" endPosition="48"
                          creatingUser="11111111-1111-1111-1111-111111111111"
                          creationDateTime="2026-03-02T10:20:00Z">
        <Coding guid="DDDDDDD1-0000-0000-0000-000000000005"
                creatingUser="11111111-1111-1111-1111-111111111111"
                creationDateTime="2026-03-02T10:20:00Z">
          <CodeRef targetGUID="AAAAAAA1-0000-0000-0000-000000000003"/>
        </Coding>
        <Coding guid="DDDDDDD1-0000-0000-0000-000000000006"
                creatingUser="11111111-1111-1111-1111-111111111111"
                creationDateTime="2026-03-02T10:21:00Z">
          <CodeRef targetGUID="AAAAAAA1-0000-0000-0000-000000000002"/>
        </Coding>
      </PlainTextSelection>
    </TextSource>
    <PDFSource guid="BBBBBBB1-0000-0000-0000-000000000003" name="Leitfaden"
               path="internal://leitfaden.pdf"
               creatingUser="11111111-1111-1111-1111-111111111111"
               creationDateTime="2026-03-01T09:07:00Z">
      <PDFSelection guid="CCCCCCC1-0000-0000-0000-000000000005" name="r1"
                    page="2" firstX="72" firstY="640" secondX="480"
                    secondY="700"
                    creatingUser="11111111-1111-1111-1111-111111111111"
                    creationDateTime="2026-03-02T10:30:00Z">
        <Coding guid="DDDDDDD1-0000-0000-0000-000000000007"
                creatingUser="11111111-1111-1111-1111-111111111111"
                creationDateTime="2026-03-02T10:30:00Z">
          <CodeRef targetGUID="AAAAAAA1-0000-0000-0000-000000000001"/>
        </Coding>
      </PDFSelection>
    </PDFSource>
    <PictureSource guid="BBBBBBB1-0000-0000-0000-000000000004"
                   name="Whiteboard" path="internal://board.png"/>
  </Sources>
  <Cases>
    <Case guid="EEEEEEE1-0000-0000-0000-000000000001" name="Station A">
      <SourceRef targetGUID="BBBBBBB1-0000-0000-0000-000000000001"/>
    </Case>
  </Cases>
  <Notes>
    <Note guid="FFFFFFF1-0000-0000-0000-000000000001" name="Memo"
          plainTextPath="internal://memo.txt"/>
  </Notes>
</Project>
"""


def build(target: pathlib.Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    stamp = (2026, 3, 1, 9, 0, 0)          # fixed, or the bytes would differ
    payload = [
        ("project.qde", QDE.format(text_b=TEXT_B).encode("utf-8")),
        ("sources/text-01.txt", TEXT_A.encode("utf-8")),
        ("sources/memo.txt", b"Erste Durchsicht, nichts Auffaelliges.\n"),
    ]
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, blob in payload:
            info = zipfile.ZipInfo(name, date_time=stamp)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, blob)


def _reference(archive: pathlib.Path) -> dict[str, object]:
    """What a correct reader must produce, frozen so three languages agree.

    Written from the Python reader because it is the one with the larger test
    suite behind it; qdaR compares against the same file, so the two are held
    to one reference rather than to each other.
    """
    sys.path.insert(0, str(HERE / "src"))
    import pandas as pd

    from qdapy.qdpx import read_qdpx

    project = read_qdpx(archive, warn=False)

    def records(df: pd.DataFrame) -> list[dict[str, object]]:
        """NaN is not valid JSON; an absent number is null."""
        clean = df.astype(object).where(df.notna(), None)
        return clean.to_dict(orient="records")

    return {
        "fragments": records(project.fragments),
        "codebook": records(project.codebook),
        "history": records(project.history),
        "uncoded": records(project.uncoded),
        "multi_coded": records(project.multi_coded),
        "coders": project.coders,
        "sources": project.sources,
        "skipped": project.skipped,
        "limitations": project.limitations,
    }


def main() -> int:
    targets = [HERE / "src" / "qdapy" / "data" / "sample.qdpx"]
    twin = HERE.parent / "qdaR" / "inst" / "extdata" / "sample.qdpx"
    if twin.parent.exists():
        targets.append(twin)
    for t in targets:
        build(t)
        print(f"wrote {t} ({t.stat().st_size} bytes)")
    ref = json.dumps(_reference(targets[0]), ensure_ascii=False, indent=1,
                     sort_keys=True) + "\n"
    ref_targets = [HERE / "tests" / "qdpx-reference.json"]
    if len(targets) == 2:
        ref_targets.append(HERE.parent / "qdaR" / "tests" / "testthat"
                           / "qdpx-reference.json")
    for t in ref_targets:
        t.write_text(ref, encoding="utf-8")
        print(f"wrote {t} ({len(ref)} bytes)")

    if len(targets) == 2:
        same = (targets[0].read_bytes() == targets[1].read_bytes()
                and ref_targets[0].read_bytes() == ref_targets[1].read_bytes())
        print("byte-identical across the two packages:", same)
        return 0 if same else 1
    print("qdaR not beside this package; wrote only the local copies")
    return 0


if __name__ == "__main__":
    sys.exit(main())
