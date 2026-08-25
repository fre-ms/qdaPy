# Documentation

Quarto is the documentation system; the `.qmd` files in `en/` and `de/`
are the reference. One project per language, exactly parallel — the
same pages in the same order, which `gen_langmap.py` checks on every
build when it derives the language switcher's page map from the two
sidebars.

    build.sh          language map -> render en + de into ../site/{en,de}
                      -> offline repairs. Pins the Quarto version; the
                      header explains how to upgrade deliberately.
    gen_langmap.py    the page-to-page map for the language switcher,
                      derived from the two _quarto.yml sidebars
    bibliography.py   builds shared/references.bib from the running
                      Zotero (Better BibTeX, DOI first). Manual and
                      minutes-slow on a large library — run it when the
                      literature changes, not on every build
    _theme/           the one vendored copy of the theme extension and
                      the Noto fonts; build.sh mirrors it into en/ and
                      de/ before rendering (Quarto resolves extensions
                      only inside a project, and Nextcloud does not
                      sync symlinks) — the per-language copies are
                      untracked build artifacts
    update-theme.sh   re-vendors _theme/, offline/ and print/ from a
                      sibling zotqda-quarto-theme checkout
    offline/          vendored from the theme: postprocess.py (makes the
                      rendered site work from a plain file tree, fails
                      loudly when Quarto's output drifts), smoketest.py
                      (headless-browser proof), mathjax/
    shared/           versions.js (version banner master) and
                      references.bib (generated)

Executable pages (`guide/figures.qmd` and its German twin) run against
the demo export that ships with the package, in a venv with jupyter and
an editable install of this checkout. It lives outside the synchronised
tree on purpose:

    python3 -m venv ~/.venvs/qdapy-docs
    ~/.venvs/qdapy-docs/bin/pip install jupyter -e .

Verify a build with

    python3 doc/offline/smoketest.py site/en site/de

Deployment is unchanged: `site/` is uploaded under a version directory,
`script/gen_versions.py` maintains `versions.json`, and the inlined
version banner does the rest.
