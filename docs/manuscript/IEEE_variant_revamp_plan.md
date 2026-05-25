# IEEE Variant Revamp Plan

Goal: Produce a strict IEEE-style manuscript variant from validated HELIX content, with proper title/author block, Index Terms, professional section naming, clean figure/table captions, and a DOCX-first publishing workflow.

Workflow constraints (requested):
1. Plan first.
2. Build DOCX variant first (not markdown->PDF direct path).
3. Review rendered PDF in browser.
4. Adjust layout/figures if cluttered.
5. Update markdown to match final doc variant.
6. Deliver final DOCX + PDF + markdown.

## Task 1: Build IEEE markdown source (content-preserving)
- Source: `docs/manuscript/HELIX_submission_ready.md`
- Target: `docs/manuscript/HELIX_ieee_variant.md`
- Changes:
  - Add IEEE-style author block and affiliation details.
  - Add `Index Terms` line after abstract.
  - Tighten section naming conventions (IEEE style wording).
  - Keep all validated formulas, metrics, tables, and claims unchanged.
  - Ensure figure captions remain in text, not embedded in images.

## Task 2: Generate IEEE-style DOCX using template
- Template: `docs/manuscript/conference-template-letter.docx`
- Command path:
  - `pandoc HELIX_ieee_variant.md -> HELIX_ieee_variant.docx --reference-doc=conference-template-letter.docx`

## Task 3: Generate PDF from DOCX (DOCX-first requirement)
- Command path:
  - `pandoc HELIX_ieee_variant.docx -> HELIX_ieee_variant.pdf --pdf-engine=tectonic`

## Task 4: Visual QA in browser
- Open generated PDF in browser tool.
- Check:
  - title/author/index terms placement,
  - table readability,
  - figure sizing and whitespace,
  - page clutter/overflow.

## Task 5: If clutter found, remediate
- Apply controlled figure sizing in markdown (`{ width=... }`) and/or regenerate layout-friendly images.
- Rebuild DOCX then PDF.
- Re-check in browser.

## Final outputs
- `docs/manuscript/HELIX_ieee_variant.md`
- `docs/manuscript/HELIX_ieee_variant.docx`
- `docs/manuscript/HELIX_ieee_variant.pdf`
