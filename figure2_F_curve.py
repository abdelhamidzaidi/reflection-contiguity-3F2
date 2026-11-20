"""
figure2_F_curve.py
==================

Reproduces Figure 2 of the manuscript: the function F(alpha) of Eq. (1) on
the interval 1/2 < alpha <= 4.25, together with the values obtained from
the evaluations discussed in the paper:

  * integers alpha = n (terminating series, Remark 1), exact rational values
    from f3f2.F_integer_exact;
  * half-integers alpha = p + 1/2 (Theorem 3.3 of the AIMS 2025 paper,
    corrected value (28) for p = 3), from f3f2.F_half_integer;
  * odd quarter-integers alpha = q/4 (Theorem 4 and Corollary 3), from
    f3f2.F_quarter_AB;
  * thirds alpha = m/3 (Proposition 2), where the value is reduced to the
    single series U_alpha(1-alpha) of Eq. (75), from f3f2.F_thirds.

The continuous curve is obtained by direct summation of the series (1). The
right panel shows the absolute difference between each lattice value and
the direct sum, on a logarithmic scale.
"""

import mpmath as mp
import numpy as np

import f3f2
import plotstyle
from plotstyle import plt

f3f2.set_precision(30)
FLOOR = 1e-32        # values below this are displayed at the floor


def lattice_values(amax=4.25):
    pts = {"int": [], "half": [], "quarter": [], "third": []}
    for n in range(1, int(amax) + 1):
        v = f3f2.F_integer_exact(n)
        pts["int"].append((mp.mpf(n), mp.mpf(v.numerator) / v.denominator))
    for p in range(1, int(amax - 0.5) + 1):
        pts["half"].append((p + mp.mpf(1) / 2, f3f2.F_half_integer(p)))
    for q in range(3, int(4 * amax) + 1, 2):
        pts["quarter"].append((mp.mpf(q) / 4, f3f2.F_quarter_AB(q)))
    for m in range(2, int(3 * amax) + 1):
        if m % 3:
            pts["third"].append((mp.mpf(m) / 3, f3f2.F_thirds(m)))
    return pts


STYLE = {"int": ("s", "integers (terminating)"),
         "half": ("D", r"half-integers (Theorem 3.3 of AIMS 2025)"),
         "quarter": ("o", r"odd quarter-integers (Theorem 4)"),
         "third": ("^", r"$\alpha=m/3$ (Proposition 2)")}


def main():
    alphas = np.linspace(0.52, 4.25, 120)
    curve = np.array([float(f3f2.F(a)) for a in alphas])
    pts = lattice_values()
    floored = {}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(plotstyle.WIDTH_IN, 0.42 * plotstyle.WIDTH_IN),
                                   gridspec_kw={"width_ratios": [1.35, 1]})
    ax1.plot(alphas, curve, color="0.55", lw=1.1, label=r"series (1)")
    for key, (mk, lab) in STYLE.items():
        x = [float(a) for a, _ in pts[key]]
        y = [float(v) for _, v in pts[key]]
        ax1.plot(x, y, mk, ms=3.6, mfc="white" if key == "third" else "k", mec="k",
                 mew=0.7, ls="none", label=lab)
        raw = [float(abs(v - f3f2.F(a))) for a, v in pts[key]]
        hi = [(xx, r) for xx, r in zip(x, raw) if r >= FLOOR]
        lo = [xx for xx, r in zip(x, raw) if r < FLOOR]
        if hi:
            ax2.semilogy([xx for xx, _ in hi], [r for _, r in hi], mk, ms=3.6,
                         mfc="white" if key == "third" else "k", mec="k", mew=0.7, ls="none")
        if lo:   # below the floor (including exact zeros): crosses at the floor
            ax2.semilogy(lo, [FLOOR] * len(lo), "x", ms=3.6, color="0.3", mew=0.7, ls="none")
        floored[key] = (len(lo), len(raw), max(raw) if raw else 0.0)
    ax1.set_xlabel(r"$\alpha$")
    ax1.set_ylabel(r"$F(\alpha)$")
    ax1.set_title(r"(a) $F(\alpha)$ and its lattice values")
    ax1.legend(frameon=False, loc="upper right", fontsize=7)
    ax2.set_xlabel(r"$\alpha$")
    ax2.set_ylabel("absolute difference")
    ax2.set_title("(b) agreement with direct summation")
    ax2.set_ylim(FLOOR / 10, 1e-20)
    fig.tight_layout(w_pad=1.5)
    for key, (nlo, ntot, mx) in floored.items():
        print(f"{key}: {nlo} of {ntot} differences below the floor {FLOOR:g}, max difference {mx:.3e}")
    plotstyle.save(fig, "figure2")
    print("saved figures/figure2.pdf")


if __name__ == "__main__":
    main()
