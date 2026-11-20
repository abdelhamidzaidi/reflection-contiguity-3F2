# Reflection, Contiguity and a Uniqueness Classification for the Hypergeometric Family 3F2(1−α, 1, 2α; α+1, 2α+1; 1)

Python code accompanying the article

> A. T. Zaidi, *Reflection, Contiguity and a Uniqueness Classification for a Functional System
> Satisfied by the Hypergeometric Family 3F2(1−α, 1, 2α; α+1, 2α+1; 1)*, submitted to *Mathematics* (MDPI).

The scripts reproduce every table, figure, numerical check and symbolic certificate of the article.
Equation, theorem and table numbers in the comments refer to the manuscript.
Appendix A of the article describes each script; Appendix B describes the certificate of Proposition 1.

Author: Abdelhamid Taieb Zaidi, Department of Mathematics, College of Science, Qassim University
(ORCID [0000-0003-1305-4959](https://orcid.org/0000-0003-1305-4959)).
To cite this repository, use the metadata in `CITATION.cff`.

## Contents

```
README.md                 this file
LICENSE                   MIT licence
CITATION.cff              citation metadata
requirements.txt          package versions used for the supplied outputs
f3f2.py, plotstyle.py     shared module (S1) and figure settings (S2)
*.py                      scripts S3-S13 (table below)
output/                   console output of every script (*.txt),
                          thomae_certificate.txt (240 condition pairs, Appendix B),
                          pslq_basis.json and pslq_result.json (Remark 10)
figures/                  figure1.pdf, figure2.pdf, figure3.pdf (Figures 1-3)
```

## Requirements

Python 3.10 or later with `mpmath`, `sympy`, `numpy` and `matplotlib`. The supplied outputs were produced with Python 3.13 and the versions pinned in `requirements.txt`.

## Exact commands

After cloning the repository, from its top-level folder:

```
git clone https://github.com/abdelhamidzaidi/reflection-contiguity-3F2.git
cd reflection-contiguity-3F2
python -m venv venv
venv/bin/pip install -r requirements.txt        # Windows: venv\Scripts\pip install -r requirements.txt
venv/bin/python run_all.py                       # Windows: venv\Scripts\python run_all.py
```

`run_all.py` runs scripts S3-S11 and S13 and rewrites `output/` and `figures/`. A single script can be run on its own, for example `venv/bin/python table3_quarter_values.py`.

Reproduction check: in a fresh virtual environment created from `requirements.txt`, `run_all.py` regenerated every file in `output/` and `figures/` byte for byte. The only exception is the line of machine-dependent timings printed by `verify_identities.py`. The printed rows agree with Tables 2, 3, 4 and B1 of the article, and the floor counts printed by the figure scripts agree with the figure captions.

## Scripts

| Script | Reproduces |
|---|---|
| `f3f2.py` | shared definitions (all formulas of the paper) and the fast summation routine `hyp_unit` |
| `plotstyle.py` | common figure settings |
| `table2_thomae_orbit.py` | Table 2, Proposition 1, Table B1 and the certificate `output/thomae_certificate.txt` (every ordering of every orbit element, its two conditions and its solution set) |
| `table3_quarter_values.py` | Table 3, Theorem 4, Corollary 3 |
| `table4_reflection_check.py` | Table 4, Corollary 1 |
| `verify_identities.py` | Section 11 and the special values (27), (28), (78), (96) |
| `pslq_search.py` | integer relation experiment of Remark 10 (120 and 240 digits); writes `output/pslq_basis.json` and `output/pslq_result.json` |
| `gosper_check.py` | Gosper certificate of Lemma 5 (Section 8) |
| `figure1_classification.py` | Figure 1 (Theorem 2) |
| `figure2_F_curve.py` | Figure 2 |
| `figure3_residuals.py` | Figure 3 |
| `verify_uniqueness.py` | Lemma 4, Remark 2, Theorem 3 and Proposition 4 (Sections 5 and 10) |
| `run_all.py` | runs all of the above |
