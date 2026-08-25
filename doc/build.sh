#!/bin/sh
# Build the documentation: regenerate the language map from the two
# sidebars, render both language projects into ../site/{en,de}, then
# apply the offline repairs so the result works from a plain file tree.
# Verify with:  python3 doc/offline/smoketest.py site/en site/de
set -e
cd "$(dirname "$0")"

# Quarto is pinned: the theme's rules and the offline repairs are
# written against this version's output. To upgrade: install the new
# Quarto, build with ALLOW_UNPINNED_QUARTO=1, run the smoketest, look
# at one site — then move the pin (and the one in CI).
PINNED="1.10.18"
ACTUAL="$(quarto --version)"
if [ "$ACTUAL" != "$PINNED" ] && [ -z "$ALLOW_UNPINNED_QUARTO" ]; then
  echo "doc/build.sh: quarto is $ACTUAL, pinned is $PINNED" >&2
  echo "  upgrades are deliberate: ALLOW_UNPINNED_QUARTO=1, smoketest," >&2
  echo "  eyeball a site, then edit PINNED here and in the CI workflow" >&2
  exit 1
fi

# Executable cells need a Python with jupyter and this package. The
# default is the local docs venv, which lives OUTSIDE the Nextcloud
# tree on purpose (the sync corrupts native-library signatures); CI
# exports its own QUARTO_PYTHON before calling this script.
export QUARTO_PYTHON="${QUARTO_PYTHON:-$HOME/.venvs/qdapy-docs/bin/python}"


# The vendored theme and the bibliography live once, in _theme/ and
# shared/; each language project gets a disposable copy before the
# render, because Quarto only resolves extensions and resources inside a
# project directory (and Nextcloud does not sync symlinks).
for lang in en de; do
  rsync -a --delete _theme/_extensions/ "$lang/_extensions/"
  rsync -a --delete _theme/fonts/ "$lang/fonts/"
  cp shared/references.bib "$lang/references.bib"
done

"$QUARTO_PYTHON" gen_langmap.py --extra-js shared/versions.js en de
quarto render en
quarto render de
python3 offline/postprocess.py ../site/en ../site/de

# The same pages once more as one linked PDF per language (Typst, Noto
# Sans embedded), downloadable via the navbar's PDF icon.
"$QUARTO_PYTHON" print/make_pdf.py en ../site/en/qdaPy-documentation.pdf
"$QUARTO_PYTHON" print/make_pdf.py de ../site/de/qdaPy-Dokumentation.pdf
echo "Done: open ../site/en/index.html"
