"""Export every notebook to PDF for the course submission.

Each notebook is converted to HTML with its saved outputs (nothing is re-run), then printed to
PDF by headless Google Chrome. A small print stylesheet wraps long code lines, and a script shrinks wide tables and
printed outputs to the page width, so nothing is cut off at the page edge. Each PDF has the same name as its notebook.

Needs Google Chrome (default macOS location below; pass --chrome to use another path).

Usage:
    conda activate stormtrack
    python scripts/export_notebook_pdfs.py              # all notebooks
    python scripts/export_notebook_pdfs.py 02_eda       # just one
Output: reports/notebook_pdfs/<notebook name>.pdf
"""
import argparse
import subprocess
import tempfile
from pathlib import Path

import nbformat
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"
OUT = ROOT / "reports" / "notebook_pdfs"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

PRINT_CSS = """
<style>
  @page { size: Letter; margin: 0.55in 0.5in; }
  body { font-size: 11px; }
  .jp-InputArea-editor pre, .jp-InputArea-editor code {
      white-space: pre-wrap !important; word-break: break-word; overflow: visible !important; }
  .jp-OutputArea-output pre { overflow: visible !important; }
  .jp-OutputArea-output, .jp-RenderedHTMLCommon { overflow: visible !important; }
  /* lay the page out at its printed width (Letter minus margins) so the fitting script measures
     the same widths the printer will use; stop output areas from stretching to fit their content */
  body, .jp-Notebook { width: 700px !important; max-width: 700px !important; margin: 0 !important; padding: 0 !important; }
  .jp-OutputArea-output { min-width: 0 !important; flex: 1 1 0 !important; }
  .jp-RenderedHTMLCommon table, .dataframe { font-size: 9px !important; max-width: none !important; }
  img { max-width: 100% !important; height: auto !important; page-break-inside: avoid; }
  .jp-Cell { page-break-inside: auto; }
  .jp-InputPrompt, .jp-OutputPrompt { font-size: 9px; }
  /* give the Out[] prompt the same fixed width on screen and in print, so tables are fitted to the
     space they will really have, and never clip an output box */
  .jp-OutputPrompt { display: block !important; flex: 0 0 64px !important; width: 64px !important; min-width: 64px !important; }
  .jp-OutputArea-child { overflow: visible !important; }
</style>
<script>
  // Shrink any table or printed text output that is wider than its output area, so the page edge
  // never cuts columns off. Printed text keeps its alignment unless it would need more than 2x shrinking,
  // in which case it wraps instead.
  // Runs shortly after load: at the load event itself the layout can still report zero widths.
  window.addEventListener("load", () => setTimeout(() => {
    document.querySelectorAll(".jp-OutputArea-output").forEach(area => {
      const child = area.closest(".jp-OutputArea-child");
      const prompt = child ? child.querySelector(".jp-OutputPrompt") : null;
      let avail = (child ? child.clientWidth : 0) - (prompt ? prompt.offsetWidth : 0) - 8;
      if (avail < 200) avail = 690;                       // layout not ready: fall back to the page width
      area.querySelectorAll("table").forEach(t => {
        // Shrinking can let wrapped cells unwrap and widen the table again, so re-measure and repeat.
        let z = 1;
        for (let k = 0; k < 8; k++) {
          const w = t.getBoundingClientRect().width;
          if (w <= avail + 1) break;
          z *= (avail / w) * 0.99;
          t.style.zoom = z.toFixed(4);
        }
      });
      area.querySelectorAll("pre").forEach(pre => {
        pre.style.whiteSpace = "pre";
        pre.style.wordBreak = "normal";
        const z = avail / pre.scrollWidth;
        if (z >= 1) return;
        if (z >= 0.5) pre.style.zoom = z.toFixed(3);
        else { pre.style.whiteSpace = "pre-wrap"; pre.style.wordBreak = "break-word"; }
      });
    });
  }, 300));
</script>
"""


def export(nb_path: Path, chrome: str, tmp: Path) -> Path:
    nb = nbformat.read(nb_path, as_version=4)
    html, _ = HTMLExporter(template_name="lab").from_notebook_node(nb, resources={"metadata": {"name": nb_path.stem}})
    html = html.replace("</head>", PRINT_CSS + "</head>", 1)
    html_path = tmp / f"{nb_path.stem}.html"
    html_path.write_text(html, encoding="utf-8")
    pdf_path = OUT / f"{nb_path.stem}.pdf"
    subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-first-run",
                    f"--user-data-dir={tmp / 'chrome-profile'}",       # throwaway profile, never the user's own
                    "--no-pdf-header-footer", "--virtual-time-budget=15000",
                    f"--print-to-pdf={pdf_path}", html_path.as_uri()],
                   check=True, capture_output=True, timeout=300)
    return pdf_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help="notebook names without .ipynb (default: all)")
    parser.add_argument("--chrome", default=CHROME)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    paths = sorted(NOTEBOOKS.glob("*.ipynb"))
    if args.names:
        paths = [p for p in paths if p.stem in args.names]
    with tempfile.TemporaryDirectory() as tmp:
        for p in paths:
            pdf = export(p, args.chrome, Path(tmp))
            print(f"{p.name:40s} -> {pdf.relative_to(ROOT)} ({pdf.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
