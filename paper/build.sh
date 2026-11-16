#!/usr/bin/env bash
# Regenerate every table and figure from the result JSONs, then compile the PDF.
# Also prints the body page count, which is what the 8-page limit applies to.
set -e
cd "$(dirname "$0")/.."
venv/bin/python experiment/make_paper_tables.py
venv/bin/python experiment/make_paper_figures.py
cd paper
tectonic -X compile main.tex --outdir . --keep-logs
python3 - <<'PY'
import pathlib
s = pathlib.Path("main.tex").read_text()
pathlib.Path("_body.tex").write_text(
    s[:s.index("{\\small\n\\bibliographystyle")] + "\n\\end{document}\n")
PY
tectonic -X compile _body.tex --outdir . --keep-logs >/dev/null 2>&1 || true
echo
echo "full document: $(grep -oE '\[[0-9]+\]' main.log | tail -1) pages"
echo "body only:     $(grep -oE '\[[0-9]+\]' _body.log | tail -1) pages   <-- the 8-page limit applies here"
rm -f _body.tex _body.pdf _body.log
echo
echo "--- audit ---"
cd ..
venv/bin/python experiment/audit_paper.py | tail -6
