"""PDF export of a review (Week 4, step 4.5): renders the Markdown report from report_md.py with fpdf2 (pure Python).

    from app.ui.report_pdf import review_pdf
    data = review_pdf(review)          # bytes of a PDF; review = GET /reviews/{id}

Handles the subset report_md writes: headings, paragraphs, bullets, block quotes, pipe tables, **bold** and *italic*.
Fonts: a Unicode TTF when one is found (REPORT_FONT, DejaVu on Linux/Docker, Arial on Windows, Arial Unicode on macOS),
else the built-in Helvetica with non-Latin-1 characters replaced (→ becomes ->, ≥ becomes >=, ...), so export never fails.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

from app.ui.report_md import review_markdown

FONT_CANDIDATES = [
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
    ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/segoeuib.ttf"),
    ("/Library/Fonts/Arial Unicode.ttf", "/Library/Fonts/Arial Unicode.ttf"),
    ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
]
ASCII = {"→": "->", "←": "<-", "≥": ">=", "≤": "<=", "–": "-", "—": "-", "·": "|", "’": "'", "‘": "'", "“": '"', "”": '"',
         "₹": "INR ", "…": "...", "•": "-", "×": "x", "✓": "yes", "✗": "no", "\u00a0": " "}


def _font_files() -> tuple[str, str] | None:
    env = os.getenv("REPORT_FONT", "").strip()
    if env and Path(env).is_file():
        return env, os.getenv("REPORT_FONT_BOLD", "").strip() or env
    try:                                                      # matplotlib ships DejaVu: a font on any machine that has it
        import matplotlib
        d = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
        FONT_CANDIDATES.append((str(d / "DejaVuSans.ttf"), str(d / "DejaVuSans-Bold.ttf")))
    except Exception:
        pass
    for reg, bold in FONT_CANDIDATES:
        if Path(reg).is_file():
            return reg, bold if Path(bold).is_file() else reg
    return None


def _inline(text: str) -> str:
    """report_md inline markup -> fpdf2 markdown: **bold** stays, *italic* -> __italic__, `code` -> plain."""
    text = text.replace("`", "")
    text = re.sub(r"(?<!\*)\*(?!\*)([^*\n]+?)(?<!\*)\*(?!\*)", r"__\1__", text)
    return text


def markdown_pdf(md: str, title: str = "Vendor risk review") -> bytes:
    from fpdf import FPDF
    from fpdf.fonts import FontFace
    fonts = _font_files()
    pdf = FPDF(format="A4")
    pdf.set_title(title)
    pdf.set_author("Vendor Risk Copilot")
    pdf.set_auto_page_break(auto=True, margin=15)
    if fonts:
        pdf.add_font("Body", "", fonts[0])
        pdf.add_font("Body", "B", fonts[1])
        pdf.add_font("Body", "I", fonts[0])
        pdf.add_font("Body", "BI", fonts[1])
        family, clean = "Body", (lambda s: s)
    else:
        family = "Helvetica"
        clean = lambda s: "".join(ASCII.get(ch, ch) for ch in s).encode("latin-1", "replace").decode("latin-1")
    pdf.add_page()
    width = pdf.w - pdf.l_margin - pdf.r_margin
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        if not line:
            pdf.ln(2)
            i += 1
            continue
        if line.startswith("|"):                               # a pipe table: header, separator, rows
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i])
                i += 1
            rows = [[clean(_inline(c.strip()).replace("\\|", "|")) for c in r.strip().strip("|").split(" | ")]
                    for r in block if not re.match(r"^\|\s*-", r)]
            ncol = max(len(r) for r in rows)
            rows = [r + [""] * (ncol - len(r)) for r in rows]
            pdf.set_font(family, size=7.5)
            pdf.set_font(family, "B", size=7.5)                # widest case: header words are bold
            pad = 2.5                                           # cell padding, mm
            need = [max((pdf.get_string_width(w.replace("**", "").replace("__", "")) for r in rows for w in r[c].split()),
                        default=4) + pad for c in range(ncol)]  # never split a word
            want = [min(70.0, max(len(r[c]) for r in rows)) for c in range(ncol)]   # longer text, more room
            spare = max(0.0, width - sum(need))
            lens = tuple(n + spare * w / (sum(want) or 1) for n, w in zip(need, want))
            if sum(need) > width:                               # too many long words: proportional to need
                lens = tuple(n * width / sum(need) for n in need)
            pdf.set_font(family, size=7.5)
            with pdf.table(width=width, col_widths=lens, text_align="LEFT", line_height=4, markdown=True,
                           headings_style=FontFace(emphasis="BOLD", fill_color=(235, 235, 235))) as t:
                for r in rows:
                    row = t.row()
                    for c in r:
                        row.cell(c)
            pdf.ln(2)
            continue
        m = re.match(r"^(#{1,3})\s+(.*)", line)
        if m:
            size = {1: 15, 2: 12, 3: 10.5}[len(m.group(1))]
            pdf.ln(2)
            pdf.set_font(family, "B", size)
            pdf.multi_cell(width, size * 0.5, clean(_inline(m.group(2)).replace("**", "")), align="L", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
        elif line.startswith("> "):
            pdf.set_font(family, "I", 9.5)
            pdf.set_x(pdf.l_margin + 4)
            pdf.multi_cell(width - 4, 5, clean(_inline(line[2:])), markdown=True, align="L", new_x="LMARGIN", new_y="NEXT")
        elif re.match(r"^\s*[-*] ", line):
            pdf.set_font(family, size=9.5)
            pdf.set_x(pdf.l_margin + 3)
            pdf.multi_cell(width - 3, 5, clean("- " + _inline(re.sub(r"^\s*[-*] ", "", line))), markdown=True, align="L",
                           new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.set_font(family, size=9.5)
            pdf.multi_cell(width, 5, clean(_inline(line)), markdown=True, align="L", new_x="LMARGIN", new_y="NEXT")
        i += 1
    return bytes(pdf.output())


def review_pdf(review: dict) -> bytes:
    name = (review.get("vendor") or {}).get("legal_name") or review.get("vendor_id") or "review"
    return markdown_pdf(review_markdown(review), title=f"Vendor risk review: {name}")
