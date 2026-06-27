# arXiv Submission Checklist

Audited against https://info.arxiv.org/help/submit/index.html and
https://info.arxiv.org/help/submit_tex.html (fetched 2026-06-19).

## Upload this file
**`arxiv-submission.tar.gz`** — the packaged source. Verified to compile from
the root with `pdflatex` twice using the included `.bbl` (no bibtex needed),
0 undefined references, 0 missing figures, 13 pages.

## Compliance: automated rules (all PASS)

| Rule | Status |
|---|---|
| Submit TeX source, not PDF-from-TeX | PASS — uploading source archive |
| Main file has `\documentclass` | PASS — `main.tex` |
| `.bbl` included, name matches main `.tex` (`main.bbl`) | PASS |
| No compiled/aux output in archive (`.pdf`, `.aux`, `.log`) | PASS — source only |
| Figures in PDF/PNG/JPEG for pdflatex (no EPS mixing) | PASS — all PDF |
| All referenced figures included (none external) | PASS — 7 figures, all present |
| Filenames use only `a-zA-Z0-9_+-.,=` | PASS |
| Figure filename case matches `\includegraphics` exactly | PASS |
| Only standard style packages (no custom `.sty`) | PASS |
| `\graphicspath{{figures/}}` resolves from root | PASS — compiles from root |
| Do NOT use `\pdfoutput` to force output | PASS — not used (figures are PDF, pdflatex auto-detected) |
| Under 50 MB | PASS — 176 KB |

## Manual steps in the arXiv web form (you must do these)

1. **Create/enable an account** with submission rights. First submission to
   `cs.IR`/`cs.CL` may require an **endorsement** from an existing arXiv author;
   your prior published work may already grant standing. Check early.
2. **Primary category:** `cs.IR` (Information Retrieval). **Cross-list:** `cs.CL`,
   `cs.LG`.
3. **License:** choose at submission (e.g. CC BY 4.0 or arXiv non-exclusive).
   Chosen in the form, not in the source.
4. **Metadata** — title, authors, abstract — entered in the form; make them match
   the paper. Paste the abstract from `main.tex` (strip LaTeX macros).
5. **Upload** `arxiv-submission.tar.gz`, then **inspect arXiv's auto-generated
   PDF** before finalizing. It is permanent once announced.

## Before you submit (author decisions — not arXiv rules)

- [ ] Confirm **authorship and order** (currently Awale, Mukherjee — comment in `main.tex`).
- [ ] Pick the **final title** (3 options commented in `main.tex`).
- [ ] Proof-read prose (Related Work + Method are first-draft).
- [ ] Optional but recommended: add the **post-hoc-binary `d_eff` control row** to
      Table `tab:effdim` to harden the mechanism claim.
- [ ] If you later target an ACL-family venue, mind their anonymity-period rule
      (no non-anonymous preprint in the ~month before that deadline). ECIR/SIGIR/CIKM
      have no such rule.

## Rebuild the package after edits
```bash
cd docs/arXiv
latexmk -pdf main.tex          # regenerates build/main.bbl + build/main.pdf
# then re-run the packaging (copies main.tex, sections/, figures/, ref.bib, build/main.bbl)
```
