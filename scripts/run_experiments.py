#!/usr/bin/env python
"""Run the complete experimental protocol and write tables / figures to --out."""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prwb.experiments import Workbench          # noqa: E402
from prwb.pipeline import load_config           # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true", help="small, fast smoke run")
    ap.add_argument("--only", default=None, help="comma list of experiments, e.g. overview,stats")
    a = ap.parse_args()
    wb = Workbench(load_config(a.config), a.out, a.quick)
    if a.only:
        for name in a.only.split(","):
            getattr(wb, "exp_" + name.strip())()
    else:
        wb.run_all()


if __name__ == "__main__":
    main()
