"""COREQ: the reporting checklist, pre-filled with what the data knows.

The 32 items are reproduced verbatim from Table 1 of Tong, Sainsbury and
Craig (2007) doi:10.1093/intqhc/mzm042. A checklist paraphrased is a
checklist a reviewer cannot match against the original.
"""

from __future__ import annotations

import pandas as pd

from ._version import __version__
from .progress import new_codes, saturation_ratio

__all__ = ["ITEMS", "SRQR_ITEMS", "coreq", "coreq_markdown",
           "srqr", "srqr_markdown"]

# item number, domain, section, name, guide question
ITEMS: tuple[tuple[str, str, str, str, str], ...] = (
    ('1', 'Research team and reflexivity', 'Personal characteristics', 'Interviewer/facilitator',
     'Which author/s conducted the interview or focus group?'),
    ('2', 'Research team and reflexivity', 'Personal characteristics', 'Credentials',
     "What were the researcher's credentials? E.g. PhD, MD"),
    ('3', 'Research team and reflexivity', 'Personal characteristics', 'Occupation',
     'What was their occupation at the time of the study?'),
    ('4', 'Research team and reflexivity', 'Personal characteristics', 'Gender',
     'Was the researcher male or female?'),
    ('5', 'Research team and reflexivity', 'Personal characteristics', 'Experience and training',
     'What experience or training did the researcher have?'),
    ('6', 'Research team and reflexivity', 'Relationship with participants', 'Relationship established',
     'Was a relationship established prior to study commencement?'),
    ('7', 'Research team and reflexivity', 'Relationship with participants', 'Participant knowledge of the interviewer',
     'What did the participants know about the researcher? e.g. personal goals, reasons for doing the research'),
    ('8', 'Research team and reflexivity', 'Relationship with participants', 'Interviewer characteristics',
     'What characteristics were reported about the interviewer/facilitator? e.g. Bias, assumptions, reasons and interests in the research topic'),
    ('9', 'Study design', 'Theoretical framework', 'Methodological orientation and Theory',
     'What methodological orientation was stated to underpin the study? e.g. grounded theory, discourse analysis, ethnography, phenomenology, content analysis'),
    ('10', 'Study design', 'Participant selection', 'Sampling',
     'How were participants selected? e.g. purposive, convenience, consecutive, snowball'),
    ('11', 'Study design', 'Participant selection', 'Method of approach',
     'How were participants approached? e.g. face-to-face, telephone, mail, email'),
    ('12', 'Study design', 'Participant selection', 'Sample size',
     'How many participants were in the study?'),
    ('13', 'Study design', 'Participant selection', 'Non-participation',
     'How many people refused to participate or dropped out? Reasons?'),
    ('14', 'Study design', 'Setting', 'Setting of data collection',
     'Where was the data collected? e.g. home, clinic, workplace'),
    ('15', 'Study design', 'Setting', 'Presence of non-participants',
     'Was anyone else present besides the participants and researchers?'),
    ('16', 'Study design', 'Setting', 'Description of sample',
     'What are the important characteristics of the sample? e.g. demographic data, date'),
    ('17', 'Study design', 'Data collection', 'Interview guide',
     'Were questions, prompts, guides provided by the authors? Was it pilot tested?'),
    ('18', 'Study design', 'Data collection', 'Repeat interviews',
     'Were repeat interviews carried out? If yes, how many?'),
    ('19', 'Study design', 'Data collection', 'Audio/visual recording',
     'Did the research use audio or visual recording to collect the data?'),
    ('20', 'Study design', 'Data collection', 'Field notes',
     'Were field notes made during and/or after the interview or focus group?'),
    ('21', 'Study design', 'Data collection', 'Duration',
     'What was the duration of the interviews or focus group?'),
    ('22', 'Study design', 'Data collection', 'Data saturation',
     'Was data saturation discussed?'),
    ('23', 'Study design', 'Data collection', 'Transcripts returned',
     'Were transcripts returned to participants for comment and/or correction?'),
    ('24', 'Analysis and findings', 'Data analysis', 'Number of data coders',
     'How many data coders coded the data?'),
    ('25', 'Analysis and findings', 'Data analysis', 'Description of the coding tree',
     'Did authors provide a description of the coding tree?'),
    ('26', 'Analysis and findings', 'Data analysis', 'Derivation of themes',
     'Were themes identified in advance or derived from the data?'),
    ('27', 'Analysis and findings', 'Data analysis', 'Software',
     'What software, if applicable, was used to manage the data?'),
    ('28', 'Analysis and findings', 'Data analysis', 'Participant checking',
     'Did participants provide feedback on the findings?'),
    ('29', 'Analysis and findings', 'Reporting', 'Quotations presented',
     'Were participant quotations presented to illustrate the themes / findings? Was each quotation identified? e.g. participant number'),
    ('30', 'Analysis and findings', 'Reporting', 'Data and findings consistent',
     'Was there consistency between the data presented and the findings?'),
    ('31', 'Analysis and findings', 'Reporting', 'Clarity of major themes',
     'Were major themes clearly presented in the findings?'),
    ('32', 'Analysis and findings', 'Reporting', 'Clarity of minor themes',
     'Is there a description of diverse cases or discussion of minor themes?'),
)


def coreq(
    fragments: pd.DataFrame | None = None,
    history: pd.DataFrame | None = None,
    codebook: pd.DataFrame | None = None,
    *,
    software: str | None = None,
) -> pd.DataFrame:
    """Build the checklist, filling in the items the exports can answer.

    Six of the thirty-two are derivable: how many documents, how many coders,
    how large the code system, whether saturation was reached, what software
    was used, and how many quotable fragments there are. The point is not
    automation but agreement -- those numbers then match what the analysis
    actually did, rather than what someone remembered while writing up.

    Note what COREQ does not ask about: there is no item for intercoder
    agreement. If you computed it, it belongs in your answer to item 24 or
    25; the checklist will not prompt you.
    """
    answers: dict[str, str] = {}
    if fragments is not None:
        docs = [d for d in fragments.get("citekey", pd.Series(dtype=str))
                .astype(str).unique() if d]
        if not docs:
            docs = list(fragments.get("title", pd.Series(dtype=str))
                        .astype(str).unique())
        answers["12"] = f"{len(docs)} documents in the export"
        coders = sorted({c for c in fragments.get("codedBy", pd.Series(dtype=str))
                         .astype(str) if c})
        if coders:
            answers["24"] = f"{len(coders)} coder(s): {', '.join(coders)}"
        quoted = int((fragments.get("text", pd.Series(dtype=str))
                      .astype(str).str.len() > 0).sum())
        answers["29"] = (f"{quoted} coded fragments are available as "
                         "quotations, each with its annotation key")
    if history is not None:
        sat = saturation_ratio(new_codes(history))
        answers["22"] = (
            f"code saturation at {sat['notation']} (Guest, Namey and Chen "
            f"2020; base {sat['base_size']}, threshold "
            f"{sat['threshold'] * 100:g} per cent)"
            if sat["notation"]
            else f"code saturation not reached ({sat['reason']})")
    if codebook is not None:
        levels = pd.to_numeric(codebook.get("level"), errors="coerce")
        depth = int(levels.max()) if levels.notna().any() else 1
        answers["25"] = (f"{len(codebook)} codes over {max(1, depth)} levels; "
                         "export the code system for the guide")
    answers["27"] = software or (
        f"zotQDA (Zotero plugin) for coding; qdaPy {__version__} for analysis")

    return pd.DataFrame([
        {"item": int(no), "domain": domain, "section": section, "name": name,
         "question": question, "answer": answers.get(no, ""),
         "filled": no in answers}
        for no, domain, section, name, question in ITEMS
    ])


def coreq_markdown(checklist: pd.DataFrame,
                   title: str = "COREQ checklist") -> str:
    """Render the checklist as Markdown, ready to paste into a submission."""
    out = [f"# {title}", "",
           "Consolidated criteria for reporting qualitative research (Tong, "
           "Sainsbury and Craig 2007, doi:10.1093/intqhc/mzm042). Answers "
           "marked *from the data* were derived from the exports; the rest "
           "are for you to complete.", ""]
    for domain in checklist["domain"].unique():
        out += [f"## {domain}", ""]
        part = checklist[checklist["domain"] == domain]
        for section in part["section"].unique():
            out += [f"### {section}", ""]
            for _, r in part[part["section"] == section].iterrows():
                out.append(f"**{r['item']}. {r['name']}** -- {r['question']}")
                out.append(f"*From the data:* {r['answer']}" if r["answer"]
                           else "*To be completed.*")
                out.append("")
    return "\n".join(out)


# The 21 standards of SRQR, verbatim from Table 1 of O'Brien, Harris,
# Beckman, Reed and Cook (2014) doi:10.1097/ACM.0000000000000388.
# item, section, name, description
SRQR_ITEMS: tuple[tuple[str, str, str, str], ...] = (
    ('S1', 'Title and abstract', 'Title',
     'Concise description of the nature and topic of the study Identifying the study as qualitative or indicating the approach (e.g., ethnography, grounded theory) or data collection methods (e.g., interview, focus group) is recommended'),
    ('S2', 'Title and abstract', 'Abstract',
     'Summary of key elements of the study using the abstract format of the intended publication; typically includes background, purpose, methods, results, and conclusions'),
    ('S3', 'Introduction', 'Problem formulation',
     'Description and significance of the problem/phenomenon studied; review of relevant theory and empirical work; problem statement'),
    ('S4', 'Introduction', 'Purpose or research question',
     'Purpose of the study and specific objectives or questions'),
    ('S5', 'Methods', 'Qualitative approach and research paradigm',
     'Qualitative approach (e.g., ethnography, grounded theory, case study, phenomenology, narrative research) and guiding theory if appropriate; identifying the research paradigm (e.g., postpositivist, constructivist/interpretivist) is also recommended; rationale'),
    ('S6', 'Methods', 'Researcher characteristics and reflexivity',
     "Researchers' characteristics that may influence the research, including personal attributes, qualifications/experience, relationship with participants, assumptions, and/or presuppositions; potential or actual interaction between researchers' characteristics and the research questions, approach, methods, results, and/or transferability"),
    ('S7', 'Methods', 'Context',
     'Setting/site and salient contextual factors; rationale'),
    ('S8', 'Methods', 'Sampling strategy',
     'How and why research participants, documents, or events were selected; criteria for deciding when no further sampling was necessary (e.g., sampling saturation); rationale'),
    ('S9', 'Methods', 'Ethical issues pertaining to human subjects',
     'Documentation of approval by an appropriate ethics review board and participant consent, or explanation for lack thereof; other confidentiality and data security issues'),
    ('S10', 'Methods', 'Data collection methods',
     'Types of data collected; details of data collection procedures including (as appropriate) start and stop dates of data collection and analysis, iterative process, triangulation of sources/methods, and modification of procedures in response to evolving study findings; rationale'),
    ('S11', 'Methods', 'Data collection instruments and technologies',
     'Description of instruments (e.g., interview guides, questionnaires) and devices (e.g., audio recorders) used for data collection; if/how the instrument(s) changed over the course of the study'),
    ('S12', 'Methods', 'Units of study',
     'Number and relevant characteristics of participants, documents, or events included in the study; level of participation (could be reported in results)'),
    ('S13', 'Methods', 'Data processing',
     'Methods for processing data prior to and during analysis, including transcription, data entry, data management and security, verification of data integrity, data coding, and anonymization/deidentification of excerpts'),
    ('S14', 'Methods', 'Data analysis',
     'Process by which inferences, themes, etc., were identified and developed, including the researchers involved in data analysis; usually references a specific paradigm or approach; rationale'),
    ('S15', 'Methods', 'Techniques to enhance trustworthiness',
     'Techniques to enhance trustworthiness and credibility of data analysis (e.g., member checking, audit trail, triangulation); rationale'),
    ('S16', 'Results/findings', 'Synthesis and interpretation',
     'Main findings (e.g., interpretations, inferences, and themes); might include development of a theory or model, or integration with prior research or theory'),
    ('S17', 'Results/findings', 'Links to empirical data',
     'Evidence (e.g., quotes, field notes, text excerpts, photographs) to substantiate analytic findings'),
    ('S18', 'Discussion', 'Integration with prior work, implications, transferability, and contribution(s) to the field',
     'Short summary of main findings; explanation of how findings and conclusions connect to, support, elaborate on, or challenge conclusions of earlier scholarship; discussion of scope of application/generalizability; identification of unique contribution(s) to scholarship in a discipline or field'),
    ('S19', 'Discussion', 'Limitations',
     'Trustworthiness and limitations of findings'),
    ('S20', 'Other', 'Conflicts of interest',
     'Potential sources of influence or perceived influence on study conduct and conclusions; how these were managed'),
    ('S21', 'Other', 'Funding',
     'Sources of funding and other support; role of funders in data collection, interpretation, and reporting'),
)


def srqr(
    fragments: pd.DataFrame | None = None,
    history: pd.DataFrame | None = None,
    codebook: pd.DataFrame | None = None,
    *,
    software: str | None = None,
) -> pd.DataFrame:
    """The other reporting standard, and the broader of the two.

    SRQR covers the whole report in 21 standards rather than COREQ's focus on
    interviews and focus groups. Reach for it when your material is not
    interview transcripts, or when the journal names it.

    **Where the agreement figure belongs.** Unlike COREQ, SRQR has a home for
    it: standard S15, techniques to enhance trustworthiness, names the audit
    trail explicitly, and a coding log is one. If you computed intercoder
    agreement, that is the standard it answers.
    """
    answers: dict[str, str] = {}
    if fragments is not None:
        docs = [d for d in fragments.get("citekey", pd.Series(dtype=str))
                .astype(str).unique() if d]
        if not docs:
            docs = list(fragments.get("title", pd.Series(dtype=str))
                        .astype(str).unique())
        segments = fragments.get("annotationKey", pd.Series(dtype=str)).nunique()
        answers["S12"] = f"{len(docs)} documents, {segments} coded segments"
        coders = sorted({c for c in fragments.get("codedBy", pd.Series(dtype=str))
                         .astype(str) if c})
        if coders:
            answers["S14"] = (f"{len(coders)} coder(s) involved: "
                              f"{', '.join(coders)}")
        quoted = int((fragments.get("text", pd.Series(dtype=str))
                      .astype(str).str.len() > 0).sum())
        answers["S17"] = (f"{quoted} coded fragments are available as "
                          "evidence, each identified by its annotation key")
    if history is not None:
        sat = saturation_ratio(new_codes(history))
        answers["S8"] = (
            f"code saturation at {sat['notation']} (Guest, Namey and Chen "
            "2020) -- note this is code saturation, not meaning saturation"
            if sat["notation"] else
            f"code saturation not reached ({sat['reason']}); state your own "
            "stopping criterion")
        answers["S15"] = (f"audit trail: the coding log records {len(history)} "
                          "events with coder and time; add your intercoder "
                          "agreement here")
    answers["S13"] = software or (
        "coding in zotQDA (Zotero plugin); "
        + (f"{len(codebook)} codes; " if codebook is not None else "")
        + f"analysis in qdaPy {__version__}")

    return pd.DataFrame([
        {"item": no, "section": section, "name": name,
         "description": description, "answer": answers.get(no, ""),
         "filled": no in answers}
        for no, section, name, description in SRQR_ITEMS
    ])


def srqr_markdown(checklist: pd.DataFrame,
                  title: str = "SRQR checklist") -> str:
    """Render the SRQR checklist as Markdown."""
    out = [f"# {title}", "",
           "Standards for Reporting Qualitative Research (O'Brien, Harris, "
           "Beckman, Reed and Cook 2014, doi:10.1097/ACM.0000000000000388). "
           "Answers marked *from the data* were derived from the exports; the "
           "rest are for you to complete.", ""]
    for section in checklist["section"].unique():
        out += [f"## {section}", ""]
        for _, r in checklist[checklist["section"] == section].iterrows():
            out.append(f"**{r['item']} {r['name']}** -- {r['description']}")
            out.append(f"*From the data:* {r['answer']}" if r["answer"]
                       else "*To be completed.*")
            out.append("")
    return "\n".join(out)
