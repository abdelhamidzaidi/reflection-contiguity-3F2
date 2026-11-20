"""
table4_reflection_check.py
==========================

Reproduces Table 4 of the manuscript: the numerical check of the reflection
formula of Corollary 1,

    F(alpha)/(2 alpha) + G(alpha)/(1 - 2 alpha) = R(alpha),       Eq. (25),

at six values of alpha outside the lattice (1/4)Z, with a working precision
of 60 decimal digits (Section 11). F, G and R are defined in Eqs. (1), (24)
and (25); they are implemented in f3f2.F, f3f2.G and f3f2.R.
"""

import mpmath as mp

import f3f2

f3f2.set_precision(60)

ALPHAS = [("3/5", mp.mpf(3) / 5), ("2/3", mp.mpf(2) / 3), ("9/10", mp.mpf(9) / 10),
          ("13/10", mp.mpf(13) / 10), ("5/3", mp.mpf(5) / 3), ("13/5", mp.mpf(13) / 5)]


def main():
    print("alpha & F(alpha) & G(alpha) & R(alpha) & residual \\\\   [Table 4]")
    for label, a in ALPHAS:
        Fa, Ga, Ra = f3f2.F(a), f3f2.G(a), f3f2.R(a)
        res = abs(Fa / (2 * a) + Ga / (1 - 2 * a) - Ra)
        print(f"${label}$ & ${mp.nstr(Fa, 15, strip_zeros=False)}$ & ${mp.nstr(Ga, 15, strip_zeros=False)}$ & ${mp.nstr(Ra, 15, strip_zeros=False)}$"
              f" & {mp.nstr(res, 3)}\\\\")


if __name__ == "__main__":
    main()
