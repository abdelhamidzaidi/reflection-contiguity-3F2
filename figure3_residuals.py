"""
figure3_residuals.py
====================

Reproduces Figure 3 of the manuscript (Section 11).

Panel (a): base-10 logarithm of the relative residual of Theorem 1, Eq. (16),
      | U_alpha(beta) + U_alpha(1-beta) - D_alpha(beta) | / max(1, |D_alpha(beta)|),
  on a grid of 16 x 20 points 0.55 <= alpha <= 3 and -1.95 <= beta <= 1.95,
  with beta kept at distance at least 0.05 from the integers. The working
  precision is 20 digits, so residuals near 10^{-20} indicate agreement to
  full precision.

Panel (b): residual of Corollary 1, Eq. (25),
      | F(alpha)/(2alpha) + G(alpha)/(1-2alpha) - R(alpha) |,
  along a grid of 80 values of alpha in (1/2, 4) that avoids the
  half-integers and integers, where the formula is not defined (Remark 1).

The series are summed with f3f2.hyp_unit (a few hundredths of a second per
evaluation at 20 digits), and the grid points are distributed over all
available processor cores with multiprocessing.Pool. Running time: about
ten seconds on a standard laptop.
"""

from multiprocessing import Pool

import mpmath as mp
import numpy as np

import f3f2
import plotstyle
from plotstyle import plt

DPS = 20
f3f2.set_precision(DPS)
FLOOR = 10.0 ** (-DPS - 2)


def _theorem1_point(ab):
    """log10 of the relative residual of Eq. (16) at one grid point (worker function).
    The residual is divided by max(1, |D_alpha(beta)|), since D_alpha(beta) is
    large when beta is close to an integer."""
    a, b = mp.mpf(ab[0]), mp.mpf(ab[1])
    f3f2.set_precision(DPS)
    scale = max(1, abs(f3f2.D(a, b)))
    return np.log10(max(float(f3f2.reflection_residual(a, b) / scale), FLOOR))


def _corollary1_point(a):
    """Residual of Eq. (25) at one value of alpha (worker function)."""
    f3f2.set_precision(DPS)
    return max(float(f3f2.corollary1_residual(mp.mpf(a))), FLOOR)


def residual_grid(pool, na=16, nb=20):
    alphas = np.linspace(0.55, 3.0, na)
    betas = np.linspace(-1.95, 1.95, nb)
    Z = np.full((nb, na), np.nan)
    pts = [(i, j) for i, b in enumerate(betas) if abs(b - round(b)) >= 0.05
           for j in range(na)]
    vals = pool.map(_theorem1_point, [(alphas[j], betas[i]) for i, j in pts])
    for (i, j), v in zip(pts, vals):
        Z[i, j] = v
    return alphas, betas, Z


def corollary_curve(pool, n=80):
    alphas = [a for a in np.linspace(0.51, 3.99, n)
              if abs(2 * a - round(2 * a)) > 0.01]      # avoid half-integers and integers
    return np.array(alphas), np.array(pool.map(_corollary1_point, alphas))


def main():
    with Pool() as pool:
        alphas, betas, Z = residual_grid(pool)
        xs, rs = corollary_curve(pool)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(plotstyle.WIDTH_IN, 0.42 * plotstyle.WIDTH_IN),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")                 # rows with beta close to an integer are omitted
    im = ax1.pcolormesh(alphas, betas, np.ma.masked_invalid(Z), cmap=cmap,
                        vmin=-DPS - 2, vmax=-DPS + 2, shading="nearest")
    cb = fig.colorbar(im, ax=ax1, pad=0.02)
    cb.set_label(r"$\log_{10}$ of relative residual of (16)")
    ax1.set_xlabel(r"$\alpha$")
    ax1.set_ylabel(r"$\beta$")
    ax1.set_title("(a) Theorem 1 on a grid")
    at_floor = rs <= FLOOR
    ax2.semilogy(xs[~at_floor], rs[~at_floor], ".", ms=2.5, color="k")
    if at_floor.any():   # below the floor (including exact zeros): crosses at the floor
        ax2.semilogy(xs[at_floor], rs[at_floor], "x", ms=2.8, color="0.3", mew=0.6)
    zf = np.argwhere(Z <= np.log10(FLOOR))
    if len(zf):          # grid values at the floor: small white dots
        ax1.plot(alphas[zf[:, 1]], betas[zf[:, 0]], "o", ms=1.6, color="white")
    n_grid = int(np.sum(~np.isnan(Z)))
    print(f"panel (a): {len(zf)} of {n_grid} grid values at the floor {FLOOR:g}")
    print(f"panel (b): {int(at_floor.sum())} of {len(rs)} residuals at the floor; "
          f"max residual {rs.max():.3e}")
    ax2.set_xlabel(r"$\alpha$")
    ax2.set_ylabel("residual of (25)")
    ax2.set_title("(b) Corollary 1")
    ax2.set_ylim(FLOOR / 10, 1e-12)
    fig.tight_layout(w_pad=1.2)
    plotstyle.save(fig, "figure3")
    print("saved figures/figure3.pdf")
    print(f"max relative residual on grid (log10): {np.nanmax(Z):.2f};"
          f"  Corollary 1 max: {rs.max():.2e}")
    # The largest residual comes from rounding: recompute it at 40 digits.
    i, j = np.unravel_index(np.nanargmax(Z), Z.shape)
    a, b = mp.mpf(alphas[j]), mp.mpf(betas[i])
    with mp.workdps(40):
        r40 = f3f2.reflection_residual(a, b) / max(1, abs(f3f2.D(a, b)))
    print(f"worst grid point alpha={float(a):.4f}, beta={float(b):.4f}:"
          f" relative residual {mp.nstr(r40, 3)} at 40 digits")


if __name__ == "__main__":
    main()
