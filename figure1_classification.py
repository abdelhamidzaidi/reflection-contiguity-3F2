"""
figure1_classification.py
=========================

Reproduces Figure 1 of the manuscript, which illustrates Theorem 2.

Panel (a): alpha = 0.7, a value with 4alpha not in Z and 3alpha not in Z.
  On the chain C_1 = 2alpha + Z (Eq. (35)) we plot U_alpha(beta), Eq. (12),
  together with V_t = U_alpha + t W for t = -0.3 and t = 0.3, where W is the
  homogeneous solution of the proof of Theorem 2 (f3f2.W_homogeneous). All
  three sequences satisfy the contiguous relation (36), the reflection
  formula (37) (W is extended to C_2 = 1 - C_1 by W(beta) = -W(1-beta)) and
  the decay (38). Hence the system (36)-(38) cannot single out U_alpha(2alpha),
  that is, F(alpha) = 2alpha U_alpha(2alpha) (Lemma 2(i)).

Panel (b): residual of the reflection formula (37) at beta = 2alpha for V_t,
      | V_t(2alpha) + V_t(1-2alpha) - D_alpha(2alpha) |,
  as a function of t. For alpha = 0.7 the residual stays at the level of the
  working precision for every t. For alpha = 3/4 (4alpha in Z) the only
  homogeneous solutions of (36) on the single class C_1 = C_2 are t/Lambda,
  and the residual grows like |t| |1/Lambda(2alpha) + 1/Lambda(1-2alpha)|,
  so t = 0 is forced: this is part (i) of Theorem 2.
"""

import mpmath as mp
import numpy as np

import f3f2
import plotstyle
from plotstyle import plt

f3f2.set_precision(30)


def panel_a(ax, alpha=mp.mpf("0.7"), ts=(-0.3, 0.3), jrange=range(-4, 9)):
    betas = [2 * alpha + j for j in jrange]
    u = np.array([float(f3f2.U(alpha, b)) for b in betas])
    w = np.array([float(f3f2.W_homogeneous(alpha, b)) for b in betas])
    x = np.array([float(b) for b in betas])
    ax.axhline(0, color="0.7", lw=0.5)
    ax.plot(x, u, "o-", color="k", ms=3.5, label=r"$U_\alpha(\beta)$")
    for t, st in zip(ts, ("s--", "^:")):
        ax.plot(x, u + t * w, st, color="0.45", ms=3, lw=0.8,
                label=rf"$U_\alpha(\beta){t:+.1f}\,W_1(\beta)$")
    x0 = float(2 * alpha)
    ax.axvline(x0, color="0.8", lw=0.6, ls="-")
    ax.annotate(r"$\beta=2\alpha$", xy=(x0, ax.get_ylim()[0]), xytext=(x0 + 0.15, -1.6), fontsize=8)
    ax.set_xlabel(r"$\beta\in 2\alpha+\mathbb{Z}$")
    ax.set_ylabel("value")
    ax.set_title(r"(a) $\alpha=0.7$: three solutions of (36)–(38)")
    ax.legend(frameon=False, loc="upper right")


FLOOR = 1e-32        # values below this are drawn at the floor with open markers


def panel_b(ax, ts=np.linspace(-1, 1, 41)):
    for alpha, style, lab in ((mp.mpf("0.7"), "o", r"$\alpha=0.7$ ($4\alpha\notin\mathbb{Z}$)"),
                              (mp.mpf(3) / 4, "s", r"$\alpha=3/4$ ($4\alpha\in\mathbb{Z}$)")):
        u2, u1 = f3f2.U(alpha, 2 * alpha), f3f2.U(alpha, 1 - 2 * alpha)
        d = f3f2.D(alpha, 2 * alpha)
        if alpha == mp.mpf(3) / 4:
            # single class: homogeneous solutions are t/Lambda on all of C_1 = C_2
            w2, w1 = 1 / f3f2.Lam(alpha, 2 * alpha), 1 / f3f2.Lam(alpha, 1 - 2 * alpha)
        else:
            w2, w1 = f3f2.W_homogeneous(alpha, 2 * alpha), f3f2.W_homogeneous(alpha, 1 - 2 * alpha)
        raw = [float(abs(u2 + t * w2 + u1 + t * w1 - d)) for t in ts]
        col = "k" if style == "o" else "0.45"
        hi = [(t, r) for t, r in zip(ts, raw) if r >= FLOOR]
        lo = [t for t, r in zip(ts, raw) if r < FLOOR]
        if hi:
            ax.semilogy([t for t, _ in hi], [r for _, r in hi], style, ms=2.8, color=col, label=lab)
        if lo:
            ax.semilogy(lo, [FLOOR] * len(lo), style, ms=2.8, mfc="white", mec=col, mew=0.6,
                        label=lab + " (below floor)")
        print(f"alpha = {mp.nstr(alpha, 3)}: {len(lo)} of {len(raw)} residuals below the floor {FLOOR:g}, "
              f"max residual {max(raw):.3e}")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"residual of (37) at $\beta=2\alpha$")
    ax.set_title(r"(b) effect of the perturbation $tW$")
    ax.set_ylim(1e-33, 10)
    ax.legend(frameon=False, loc="center right", fontsize=6.5)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(plotstyle.WIDTH_IN, 0.42 * plotstyle.WIDTH_IN))
    panel_a(axes[0])
    panel_b(axes[1])
    fig.tight_layout(w_pad=1.5)
    plotstyle.save(fig, "figure1")
    print("saved figures/figure1.pdf")


if __name__ == "__main__":
    main()
