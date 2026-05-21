# docs/

Documentation and reference materials for the thesis. This directory is **not** the thesis itself — it holds the LaTeX source, formatting guidance, and prior-work references used while writing.

---

## What's where

### `report/` — your thesis (work here)
The active LaTeX project for the thesis. Entry point is `main.tex`.

```
report/
├── main.tex                  # Root file — fill in title, name, degree, committee
├── ref.bib                   # Bibliography (BibTeX)
├── Chapters/
│   ├── ch1.tex               # Chapter 1 (Introduction)
│   ├── ch2.tex               # Chapter 2
│   └── Ch3.tex               # Chapter 3
├── FrontMatter/
│   ├── preamble.tex          # LaTeX packages and global settings (do not edit lightly)
│   ├── AbstractPage.tex      # Abstract
│   ├── Acknowledgements.tex  # Acknowledgements
│   ├── epigraphOptional.tex  # Optional epigraph page
│   ├── List of Symbols Glossaries.tex  # Symbol/abbreviation definitions
│   └── Do Not Edit/          # Auto-generated front matter — leave alone
├── Back Matter/
│   └── Appendix A.tex        # Appendix
├── Figures/                  # Images referenced in the report
└── build/                    # LaTeX build artifacts (ignored by git) — compiled PDF is here
    └── main.pdf
```

To build: open in VS Code with the LaTeX Workshop extension (settings are in `.vscode/settings.json`) or run `latexmk` from the `report/` directory (configured via `.latexmkrc`).

---

### `uah_report_template/` — original UAH template (read-only reference)
The unmodified UAH Graduate School LaTeX template that `report/` was derived from. Consult this to recover original placeholder text, check template defaults, or see how a section was originally structured before edits. Do not write to this directory.

Structure mirrors `report/` exactly.

---

### `past_thesis_report/` — prior theses from the lab (reference only)
PDF copies of completed theses from previous students in the group. Useful for seeing how others structured chapters, worded sections, or formatted figures and tables under the same UAH requirements.

| File | Author |
|------|--------|
| `Bibek Panthi.pdf` | Bibek Panthi |
| `Binita Gyawali.pdf` | Binita Gyawali |
| `MD Sazzad Hossen.pdf` | MD Sazzad Hossen |
| `Nishan Pantha.pdf` | Nishan Pantha |
| `Oluwafolahanmi Adedamola Aluko.pdf` | Oluwafolahanmi Adedamola Aluko |
| `Paridhi Parajuli.pdf` | Paridhi Parajuli |
| `Rajashree Dahal.pdf` | Rajashree Dahal |
| `Sameer Gopali.pdf` | Sameer Gopali |

---

### `UAH Graduate School Thesis, Dissertation, and DNP Project Manual.pdf` — official style guide
The authoritative UAH Graduate School manual. Go here first for questions about:
- Required page order and formatting rules
- Margin, font, and spacing requirements
- Submission deadlines and procedures
- What counts as front matter vs. back matter

---

### `presentation/` — slides
Presentation materials (e.g., defense slides). Currently empty or in progress.

---

## Quick lookup

| Question | Where to look |
|----------|--------------|
| How do I add a chapter? | `report/Chapters/` — copy an existing `.tex` file and `\include` it in `main.tex` |
| How do I add a symbol/abbreviation? | `report/FrontMatter/List of Symbols Glossaries.tex` |
| Where is the compiled PDF? | `report/build/main.pdf` |
| What are the UAH formatting rules? | `UAH Graduate School Thesis, Dissertation, and DNP Project Manual.pdf` |
| How did a past student structure their lit review? | `past_thesis_report/*.pdf` |
| What did the original template look like? | `uah_report_template/` |
