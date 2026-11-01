#!/usr/bin/env python
"""Build the project report (PDF) from the Markdown chapters in report/chapters and the experiment results.

    python report/build_report.py            # -> report/PRWB_Project_Report.pdf

* chapters are Jinja2 templates (delimiters << >> and <% %>) so that every number, table and figure is taken from
  results/ (tables/*.csv, json/summary.json, figures/*.png) and docs/screenshots/*.png;
* cover-page fields come from report/meta.yaml;
* pandoc converts Markdown -> LaTeX, XeLaTeX compiles it (three passes for TOC / list of figures).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import jinja2
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "report"
RES = Path(os.environ.get("PRWB_RESULTS", ROOT / "results"))
BUILD = REP / "build"
sys.path.insert(0, str(REP))
from references import REFS  # noqa: E402

ORDER = ["00_abstract.md", "01_introduction.md", "02_literature_survey.md", "03_methodology.md",
         "04_implementation.md", "05_results.md", "06_analysis.md", "07_conclusion.md", "08_references.md",
         "09_appendix.md"]

# Algorithms written from scratch in this project (used for the "n_scratch" statistic and the appendix)
SCRATCH_ALGOS = {
    "Feature descriptors": ["HOG (L2-Hys blocks)", "LBP (default / u2 / riu2)", "GLCM (Haralick) statistics", "Hu moment invariants",
                            "Zoning (average pooling)", "Projection profiles", "Intensity statistics", "Time-domain statistics",
                            "Hjorth parameters", "Spectral descriptors", "Auto-correlation descriptors"],
    "Transforms and selection": ["Standardiser", "PCA (thin SVD)", "Fisher LDA (generalised eigenproblem)", "Fisher score",
                                 "mRMR selection", "Correlation pruning"],
    "Classifiers": ["k-NN (4 metrics)", "Gaussian Bayes (QDA / LDA / naive)", "Minimum-distance classifier",
                    "Parzen-window classifier", "Softmax regression (Adam)", "MLP (back-prop, dropout, early stopping)"],
    "Clustering": ["k-means++ (Lloyd)", "Gaussian mixture by EM", "Fuzzy c-means", "Gap statistic"],
    "Statistics and evaluation": ["Benjamini-Hochberg FDR", "Bonferroni correction", "ANOVA effect sizes (eta2, omega2)",
                                  "Bhattacharyya distance", "Mahalanobis outlier detection", "Expected calibration error",
                                  "McNemar test", "Corrected resampled t-test", "Friedman-Nemenyi procedure",
                                  "Bootstrap confidence intervals", "Hungarian cluster accuracy / purity", "Bootstrap cluster stability"],
}

# --------------------------------------------------------------------------- helpers
_cited: list[str] = []


def esc(s) -> str:
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\^{}")]:
        s = s.replace(a, b)
    return s


def cap(s: str) -> str:
    """Caption text: escape the characters that would break LaTeX but keep $math$ and \\commands usable."""
    for a, b in [("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_")]:
        s = s.replace(a, b)
    return s


def cite(*keys: str) -> str:
    for k in keys:
        if k not in REFS:
            raise KeyError(f"unknown reference key: {k}")
        if k not in _cited:
            _cited.append(k)
    return r"\cite{" + ",".join(keys) + "}"


def raw(tex: str) -> str:
    return "\n```{=latex}\n" + tex + "\n```\n"


def find_image(name: str) -> Path:
    for d in (RES / "figures", ROOT / "docs" / "screenshots", ROOT / "docs" / "figures"):
        if (d / name).exists():
            return d / name
    raise FileNotFoundError(name)


def fig(name: str, caption: str, width: float = 0.9, label: str | None = None) -> str:
    label = label or "fig:" + Path(name).stem
    p = find_image(name)
    return raw(rf"""\begin{{figure}}[!htbp]
\centering
\includegraphics[width={width}\textwidth,height=0.72\textheight,keepaspectratio]{{{p}}}
\caption{{{cap(caption)}}}\label{{{label}}}
\end{{figure}}""")


def figrow(items: list[tuple[str, str]], caption: str, label: str, width: float | None = None) -> str:
    n = len(items)
    w = width or round(0.96 / n - 0.01, 3)
    body = []
    for name, sub in items:
        body.append(rf"""\begin{{subfigure}}[t]{{{w}\textwidth}}\centering
\includegraphics[width=\textwidth]{{{find_image(name)}}}
\caption{{{cap(sub)}}}
\end{{subfigure}}""")
    return raw("\\begin{figure}[!htbp]\n\\centering\n" + "\\hfill\n".join(body) + f"\n\\caption{{{cap(caption)}}}\\label{{{label}}}\n\\end{{figure}}")


def ref(label: str) -> str:
    kind = {"fig": "Figure", "tab": "Table", "eq": "Equation", "sec": "Section", "ch": "Chapter", "app": "Appendix"}[label.split(":")[0]]
    return rf"`{kind}~\ref{{{label}}}`{{=latex}}"


def eq(tex: str, label: str | None = None) -> str:
    lab = rf"\label{{{label}}}" if label else ""
    return raw(rf"\begin{{equation}}{lab}" + "\n" + tex.strip() + "\n" + r"\end{equation}")


def fmt_cell(v, fmt=None):
    if isinstance(v, str):
        return esc(v)
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "--"
    if fmt:
        return fmt.format(v)
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int) or (hasattr(v, "dtype") and "int" in str(v.dtype)):
        return f"{int(v):,}".replace(",", "{,}") if abs(int(v)) >= 10000 else str(int(v))
    if isinstance(v, float):
        return f"{v:.3f}"
    return esc(v)


def tbl(df: pd.DataFrame, caption: str, label: str, fmts: dict | None = None, colfmt: str | None = None,
        resize: bool = False, font: str = "small", long: bool = False, bold_max: list[str] | None = None) -> str:
    fmts = fmts or {}
    cols = list(df.columns)
    ncol = len(cols)
    colfmt = colfmt or ("l" + "r" * (ncol - 1))
    best = {}
    for c in (bold_max or []):
        best[c] = df[c].max()
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            t = fmt_cell(r[c], fmts.get(c))
            if c in best and r[c] == best[c]:
                t = r"\textbf{" + t + "}"
            cells.append(t)
        rows.append(" & ".join(cells) + r" \\")
    head = " & ".join(r"\textbf{" + esc(c) + "}" for c in cols) + r" \\"
    if long:
        tex = (rf"\begin{{{font}}}" + "\n" + rf"\begin{{longtable}}{{{colfmt}}}" + "\n" + rf"\caption{{{cap(caption)}}}\label{{{label}}}\\" + "\n"
               + r"\toprule" + "\n" + head + "\n" + r"\midrule\endfirsthead" + "\n" + r"\toprule" + "\n" + head + "\n" + r"\midrule\endhead" + "\n"
               + "\n".join(rows) + "\n" + r"\bottomrule" + "\n" + r"\end{longtable}" + "\n" + rf"\end{{{font}}}")
        return raw(tex)
    inner = (rf"\begin{{tabular}}{{{colfmt}}}" + "\n" + r"\toprule" + "\n" + head + "\n" + r"\midrule" + "\n" + "\n".join(rows) + "\n"
             + r"\bottomrule" + "\n" + r"\end{tabular}")
    if resize:
        inner = r"\resizebox{\textwidth}{!}{" + inner + "}"
    return raw(rf"\begin{{table}}[!htbp]" + "\n" + r"\centering" + rf"\{font}" + "\n" + rf"\caption{{{cap(caption)}}}\label{{{label}}}" + "\n" + inner + "\n" + r"\end{table}")


def csv(name: str) -> pd.DataFrame:
    return pd.read_csv(RES / "tables" / name)


def refs_block() -> str:
    items = []
    for k in _cited:
        t = esc(REFS[k])
        t = re.sub(r"\*(.+?)\*", r"\\textit{\1}", t)
        t = re.sub(r'"(.*?)"', "\u201c\\1\u201d", t)
        items.append(rf"\bibitem{{{k}}} {t}")
    return raw("{\\small\n\\begin{thebibliography}{99}\n" + "\n\n".join(items) + "\n\\end{thebibliography}\n}")


def _count_lines(patterns) -> int:
    n = 0
    for pat in patterns:
        for f in ROOT.glob(pat):
            if "__pycache__" not in str(f):
                n += sum(1 for _ in f.open(encoding="utf-8"))
    return n


def project_stats() -> dict:
    """Facts about the code base that the report quotes (computed live, so they can never go stale)."""
    import importlib
    loc = {"package": _count_lines(["prwb/*.py", "prwb/*/*.py"]), "tests": _count_lines(["tests/*.py"]),
           "scripts": _count_lines(["scripts/*.py", "docs/*.py", "notebooks/*.py"]), "app": _count_lines(["app/*.py"]),
           "report": _count_lines(["report/*.py"])}
    loc["total"] = loc["package"] + loc["tests"] + loc["scripts"] + loc["app"]
    loc_by_module = {str(f.relative_to(ROOT)): sum(1 for _ in f.open(encoding="utf-8"))
                     for f in sorted(ROOT.glob("prwb/**/*.py")) if "__pycache__" not in str(f)}
    r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-o", "addopts=", "-q"], cwd=ROOT, capture_output=True, text=True)
    m = re.search(r"(\d+) tests? collected", r.stdout) or re.search(r"(\d+) tests?", r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "")
    n_tests = int(m.group(1)) if m else 0
    env = {"Python": sys.version.split()[0]}
    for mod, label in [("numpy", "NumPy"), ("scipy", "SciPy"), ("sklearn", "scikit-learn"), ("pandas", "pandas"), ("matplotlib", "Matplotlib"),
                       ("yaml", "PyYAML"), ("joblib", "joblib"), ("PIL", "Pillow"), ("streamlit", "Streamlit"), ("pytest", "pytest")]:
        try:
            env[label] = getattr(importlib.import_module(mod), "__version__", "n/a")
        except Exception:
            env[label] = "n/a"
    n_scratch = sum(len(v) for v in SCRATCH_ALGOS.values())
    cls = RES / "tables" / "classification_results.csv"
    n_cells = int(len(pd.read_csv(cls))) if cls.exists() else 0
    return {"loc": loc, "loc_by_module": loc_by_module, "n_tests": n_tests, "env": env, "n_scratch": n_scratch,
            "n_cls_cells": n_cells, "scratch": SCRATCH_ALGOS}


def load_summary():
    p = RES / "json" / "summary.json"
    S = json.loads(p.read_text()) if p.exists() else {}
    S.update(project_stats())
    return S


def loc_df(S: dict) -> pd.DataFrame:
    """Lines of code grouped by component (for the implementation chapter)."""
    groups: dict[str, list[int]] = {}
    for path, n in S["loc_by_module"].items():
        parts = path.split("/")
        key = "prwb/" + parts[1] + "/" if len(parts) > 2 else "prwb/ (top level)"
        groups.setdefault(key, []).append(n)
    rows = [[k, len(v), sum(v)] for k, v in sorted(groups.items())]
    for label, pat in [("tests/", "tests/*.py"), ("app/ (dashboard)", "app/*.py"), ("scripts/, docs/, notebooks/", None)]:
        if pat:
            fs = [f for f in ROOT.glob(pat)]
        else:
            fs = [f for q in ("scripts/*.py", "docs/*.py", "notebooks/*.py") for f in ROOT.glob(q)]
        rows.append([label, len(fs), sum(sum(1 for _ in f.open(encoding="utf-8")) for f in fs)])
    df = pd.DataFrame(rows, columns=["Component", "Files", "Lines"])
    df.loc[len(df)] = ["Total", int(df.Files.sum()), int(df.Lines.sum())]
    return df


def env_df(S: dict) -> pd.DataFrame:
    role = {"Python": "Implementation language", "NumPy": "Array computing; all from-scratch algorithms", "SciPy": "Linear algebra, statistics tests, FFT, optimisation",
            "scikit-learn": "Reference models, splitters, metrics, estimator API", "pandas": "Result tables", "Matplotlib": "All figures and diagrams",
            "PyYAML": "Experiment configuration", "joblib": "Parallel evaluation, model persistence", "Pillow": "Reading user-supplied digit pictures",
            "Streamlit": "Interactive dashboard", "pytest": "Automated tests"}
    return pd.DataFrame([[k, v, role[k]] for k, v in S["env"].items()], columns=["Component", "Version", "Role"])


def scratch_df(S: dict) -> pd.DataFrame:
    return pd.DataFrame([[k, ", ".join(v)] for k, v in S["scratch"].items()], columns=["Category", "Algorithms implemented from scratch"])


def stats_df(S: dict) -> pd.DataFrame:
    """One row per analysed feature space, from summary.json (statistical-analysis experiment)."""
    rows = []
    for k, v in S["stats"].items():
        rows.append([k.replace("_", " / ", 1).replace("_", " "), v["n_features"], 100 * v["frac_normal_shapiro"],
                     100 * v["n_sig_bh_5pct"] / v["n_tested"], v["median_eta2"], v["max_eta2"], v["top_feature"],
                     v["mean_abs_corr"], int(v["pairs_corr_gt_0.9"]), int(v["n_mahalanobis_outliers"]), v["hardest_pair"]])
    return pd.DataFrame(rows, columns=["Feature space", "d", "normal %", "signif. %", "median eta2", "max eta2", "top feature",
                                       "mean abs r", "pairs r>0.9", "outliers", "hardest pair"])


def pct(x, n: int = 1) -> str:
    """Format a fraction as a percentage string without the sign (0.9944 -> '99.4')."""
    return f"{100 * float(x):.{n}f}"


_CLS = {}


def cell(dataset: str, feature_set: str, classifier: str, col: str = "accuracy"):
    """Look up one value of classification_results.csv."""
    if "df" not in _CLS:
        _CLS["df"] = csv("classification_results.csv").set_index(["dataset", "feature_set", "classifier"])
    return _CLS["df"].loc[(dataset, feature_set, classifier), col]


def readfile(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8").rstrip("\n")


def source_of(rel: str, name: str) -> str:
    """Source text of a top-level function / class (or Class.method) from a Python file, for the appendix listings."""
    import ast
    text = (ROOT / rel).read_text(encoding="utf-8")
    tree = ast.parse(text)
    node = tree
    for part in name.split("."):
        node = next(n for n in ast.walk(node) if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == part)
    start = node.lineno - 1 - (len(node.decorator_list) if hasattr(node, "decorator_list") else 0)
    import textwrap
    return textwrap.dedent("\n".join(text.splitlines()[start:node.end_lineno]))


def render_chapters() -> str:
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(REP / "chapters")), variable_start_string="<<",
                             variable_end_string=">>", block_start_string="<%", block_end_string="%>",
                             comment_start_string="<#", comment_end_string="#>", undefined=jinja2.StrictUndefined,
                             trim_blocks=True, lstrip_blocks=True)
    meta = yaml.safe_load((REP / "meta.yaml").read_text())
    ctx = dict(S=load_summary(), meta=meta, cite=cite, fig=fig, figrow=figrow, tbl=tbl, csv=csv, ref=ref, eq=eq, raw=raw,
               pd=pd, esc=esc, refs_block=refs_block, RES=RES, loc_df=loc_df, env_df=env_df, scratch_df=scratch_df, stats_df=stats_df, pct=pct, cell=cell, readfile=readfile, source_of=source_of)
    parts = []
    for fn in ORDER:
        if not (REP / "chapters" / fn).exists():
            continue
        parts.append(env.get_template(fn).render(**ctx))
    uncited = [k for k in REFS if k not in _cited]
    if uncited:
        print("WARNING: references defined but never cited:", uncited)
    return "\n\n".join(parts)


def front_matter(meta: dict) -> str:
    authors = " \\\\ ".join(esc(a["name"]) for a in meta["authors"])
    rows = "\n".join(rf"{esc(a['name'])} & {esc(a['roll'])} \\" for a in meta["authors"])
    return rf"""
\begin{{titlepage}}
\thispagestyle{{empty}}
\begin{{center}}
{{\Large\bfseries\sffamily {esc(meta['institution'])}}}\\[4pt]
{{\large {esc(meta['university'])}}}\\[4pt]
{{\large\sffamily {esc(meta['department'])}}}\\[1.2cm]
{{\color{{accent}}\rule{{\textwidth}}{{1.2pt}}}}\\[0.9cm]
{{\LARGE\bfseries\sffamily\MakeUppercase{{{esc(meta['report_type'])}}}}}\\[0.5cm]
{{\large on}}\\[0.5cm]
{{\fontsize{{26}}{{32}}\selectfont\bfseries\sffamily {esc(meta['title'])}}}\\[0.45cm]
{{\large\itshape {esc(meta['subtitle'])}}}\\[0.8cm]
{{\color{{accent}}\rule{{\textwidth}}{{1.2pt}}}}\\[1.0cm]
{{\normalsize Submitted in partial fulfilment of the requirements of the course}}\\[2pt]
{{\large\bfseries {esc(meta['course'])}}}\\[2pt]
{{\normalsize for the degree of {esc(meta['degree'])} in {esc(meta['branch'])}}}\\[1.3cm]
\begin{{minipage}}[t]{{0.52\textwidth}}
\raggedright{{\bfseries\sffamily Submitted by}}\\[6pt]
\begin{{tabular}}{{@{{}}l l@{{}}}}
{rows}
\end{{tabular}}
\end{{minipage}}\hfill
\begin{{minipage}}[t]{{0.40\textwidth}}
\raggedright{{\bfseries\sffamily Under the guidance of}}\\[6pt]
{esc(meta['guide'])}\\[0pt]
{esc(meta['guide_designation'])}
\end{{minipage}}\\[1.6cm]
{{\large\bfseries Academic Year {esc(meta['academic_year'])}}}\\[3pt]
{{\normalsize {esc(meta['submission'])}}}
\end{{center}}
\end{{titlepage}}
\pagenumbering{{roman}}
\setcounter{{page}}{{1}}
\renewcommand{{\contentsname}}{{Table of Contents}}
\tableofcontents
\clearpage
\listoffigures
\clearpage
\listoftables
\clearpage
"""


def main():
    BUILD.mkdir(exist_ok=True)
    meta = yaml.safe_load((REP / "meta.yaml").read_text())
    md = render_chapters()
    (BUILD / "report.md").write_text(md)
    (BUILD / "before.tex").write_text(front_matter(meta))
    cmd = ["pandoc", str(BUILD / "report.md"), "-f", "markdown+raw_tex+tex_math_dollars+pipe_tables+smart-auto_identifiers",
           "-t", "latex", "-s", "--top-level-division=chapter", "--number-sections", "--wrap=none",
           "-V", "documentclass=report", "-V", "fontsize=12pt", "-V", "classoption=oneside", "-V", "papersize=a4",
           "-V", "linkcolor=accent", "--highlight-style=tango",
           "-H", str(REP / "header.tex"), "-B", str(BUILD / "before.tex"), "-o", str(BUILD / "report.tex")]
    subprocess.run(cmd, check=True)
    # pandoc's template loads lmodern, which XeLaTeX + fontspec does not need; stub it if the TeX install lacks it
    if subprocess.run(["kpsewhich", "lmodern.sty"], capture_output=True, text=True).stdout.strip() == "":
        (BUILD / "lmodern.sty").write_text("\\ProvidesPackage{lmodern}\n")
    for i in range(3):
        r = subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error", "report.tex"], cwd=BUILD,
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-3500:])
            sys.exit("xelatex failed")
    out = REP / "PRWB_Project_Report.pdf"
    shutil.copy(BUILD / "report.pdf", out)
    try:
        from pypdf import PdfReader
        print(f"wrote {out}  ({len(PdfReader(str(out)).pages)} pages, {out.stat().st_size / 1e6:.2f} MB)")
    except ImportError:
        print("wrote", out)


if __name__ == "__main__":
    main()
