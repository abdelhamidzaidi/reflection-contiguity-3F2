"""
run_all.py
==========

Runs every supplementary script in the order in which the corresponding
results appear in the manuscript, and writes the console output of each
script to the folder ./output. The three figures are written as PDF files
to figures.

Approximate total running time on a standard laptop: a few minutes. The
nested quadratures in verify_identities.py and verify_uniqueness.py account
for most of it.
"""

import contextlib
import importlib
import io
import os
import time

SCRIPTS = [
    "table2_thomae_orbit",      # Table 2 and Proposition 1
    "figure1_classification",   # Figure 1 (Theorem 2)
    "table3_quarter_values",    # Table 3 (Theorem 4, Corollary 3)
    "figure2_F_curve",          # Figure 2
    "pslq_search",              # Remark 10
    "gosper_check",             # Section 8
    "table4_reflection_check",  # Table 4 (Corollary 1)
    "verify_identities",        # Section 11 and the special values of the text
    "figure3_residuals",        # Figure 3
    "verify_uniqueness",        # Sections 5 and 10 (Lemma 4, Theorem 3, Proposition 4)
]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "output")
    os.makedirs(out_dir, exist_ok=True)
    for name in SCRIPTS:
        start = time.time()
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            importlib.import_module(name).main()
        with open(os.path.join(out_dir, name + ".txt"), "w", encoding="utf-8") as fh:
            fh.write(buffer.getvalue())
        print(f"{name:<28s} done in {time.time() - start:6.1f} s")


if __name__ == "__main__":
    main()
