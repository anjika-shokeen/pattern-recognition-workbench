"""Produce the screenshots used in the project report (docs/screenshots/*.png).

* terminal sessions: the real CLI output is captured and rendered in a terminal-style frame
* dashboard: the Streamlit app is started headless and driven with Playwright / Chromium

Usage:  python docs/make_screenshots.py [--only terminal|app]
"""
import argparse
import html
import os
import re
import shlex
import socket
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "docs", "screenshots")
os.makedirs(OUT, exist_ok=True)

TERMINALS = {
    "term_datasets": "python -m prwb datasets",
    "term_features": "python -m prwb features -d digits -f hog+lbp",
    "term_stats": "python -m prwb stats -d wine -f raw --top 10",
    "term_classify": "python -m prwb classify -d digits -f hog+lbp -c SVM-RBF --folds 5",
    "term_cluster": "python -m prwb cluster -d digits -f hog+lbp -a Agglomerative-Ward",
    "term_pytest": "python -m pytest -q",
}

PAGE = """<html><body style="margin:0;background:#fff"><div id="t" style="display:inline-block;background:#1e1f26;border-radius:8px;
overflow:hidden;box-shadow:0 2px 8px #0003;font-family:'DejaVu Sans Mono',monospace;font-size:13px;min-width:760px;max-width:1100px">
<div style="background:#2d2f3a;padding:7px 12px;display:flex;gap:7px;align-items:center">
<span style="width:11px;height:11px;border-radius:50%;background:#ff5f56"></span><span style="width:11px;height:11px;border-radius:50%;background:#ffbd2e"></span>
<span style="width:11px;height:11px;border-radius:50%;background:#27c93f"></span><span style="color:#aab;margin-left:10px;font-size:12px">prwb - terminal</span></div>
<pre style="margin:0;padding:14px 16px;color:#d7dae0;line-height:1.38;white-space:pre-wrap">{body}</pre></div></body></html>"""


def run(cmd):
    p = subprocess.run(shlex.split(cmd), cwd=ROOT, capture_output=True, text=True,
                       env={**os.environ, "PYTHONWARNINGS": "ignore", "COLUMNS": "110"})
    return (p.stdout + p.stderr).rstrip()


def terminals():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as pw:
        br = pw.chromium.launch()
        pg = br.new_page(device_scale_factor=2)
        for name, cmd in TERMINALS.items():
            t0 = time.time()
            out = run(cmd)
            lines = out.splitlines()
            if len(lines) > 42:                       # keep screenshots readable
                lines = lines[:40] + ["...", lines[-1]]
            body = f'<span style="color:#7ee787">$</span> <span style="color:#fff">{html.escape(cmd)}</span>\n' + html.escape("\n".join(lines))
            body = re.sub(r"(\d+ passed[^<]*)", r'<span style="color:#7ee787">\1</span>', body)
            pg.set_content(PAGE.format(body=body))
            pg.locator("#t").screenshot(path=os.path.join(OUT, name + ".png"))
            print(f"  {name}.png ({time.time() - t0:.0f}s)")
        br.close()


def free_port():
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def dashboard():
    from playwright.sync_api import sync_playwright
    port = free_port()
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py", "--server.headless", "true",
                             "--server.port", str(port), "--browser.gatherUsageStats", "false"], cwd=ROOT,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(8)
        with sync_playwright() as pw:
            br = pw.chromium.launch()
            pg = br.new_page(viewport={"width": 1500, "height": 1000}, device_scale_factor=1.5)
            pg.goto(f"http://localhost:{port}", wait_until="networkidle")
            pg.wait_for_selector("text=Data & features", timeout=90000)
            time.sleep(6)
            pg.screenshot(path=os.path.join(OUT, "app_data.png"), full_page=True)
            print("  app_data.png")
            pg.get_by_role("tab", name="Statistics").click(); time.sleep(7)
            pg.screenshot(path=os.path.join(OUT, "app_stats.png"), full_page=True); print("  app_stats.png")
            pg.get_by_role("tab", name="Classification").click(); time.sleep(2)
            pg.get_by_role("button", name="Run cross-validation").click(); time.sleep(25)
            pg.screenshot(path=os.path.join(OUT, "app_classification.png"), full_page=True); print("  app_classification.png")
            pg.get_by_role("tab", name="Clustering").click(); time.sleep(4)
            pg.get_by_role("button", name="Cluster", exact=True).click(timeout=120000); time.sleep(15)
            pg.screenshot(path=os.path.join(OUT, "app_clustering.png"), full_page=True); print("  app_clustering.png")
            # recognise a digit: upload a real test image
            import numpy as np
            from PIL import Image
            sys.path.insert(0, ROOT)
            from prwb import datasets as D
            ds = D.load_dataset("digits")
            i = int(np.flatnonzero(ds.y == 5)[4])
            png = os.path.join(OUT, "_upload_digit.png")
            big = np.kron(ds.X[i], np.ones((12, 12)))
            Image.fromarray((255 * (1 - big)).astype(np.uint8)).save(png)
            pg.get_by_role("tab", name="Recognise a digit").click(timeout=120000); time.sleep(4)
            pg.set_input_files("input[type=file]", png); time.sleep(15)
            pg.screenshot(path=os.path.join(OUT, "app_recognise.png"), full_page=True); print("  app_recognise.png")
            os.remove(png)
            br.close()
    finally:
        proc.terminate()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["terminal", "app"])
    a = ap.parse_args()
    if a.only in (None, "terminal"):
        terminals()
    if a.only in (None, "app"):
        dashboard()
