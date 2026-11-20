"""
table3_quarter_values.py
========================

Reproduces Table 3 of the manuscript: the explicit values

    F(q/4) = A_q + B_q kappa_q,     q = 3, 5, 7, 9, 11,          Eq. (68),

of Theorem 4 and Corollary 3 (Section 6).

For each odd q the script
  * computes A_q exactly (finite sum of Eq. (61)) and B_q exactly (proof of
    Corollary 3), using rational arithmetic in f3f2.A_q_exact / f3f2.B_q_exact;
  * evaluates the closed form (68) and the Gamma form (60) at 50 digits;
  * compares both with a direct summation of the series (1);
  * prints the corresponding LaTeX row of Table 3.
"""

import mpmath as mp

import f3f2

f3f2.set_precision(50)


def main(q_values=(3, 5, 7, 9, 11)):
    print("q & A_q & B_q & kappa_q & F(q/4) \\\\   [Table 3]")
    for q in q_values:
        A, B = f3f2.A_q_exact(q), f3f2.B_q_exact(q)
        kap = r"\Gamma(1/4)^{4}/\pi" if q % 4 == 1 else r"\pi^{3}/\Gamma(1/4)^{4}"
        closed = f3f2.F_quarter_AB(q)          # Eq. (68)
        gamma_form = f3f2.F_quarter_gamma(q)   # Eq. (60)
        direct = f3f2.F(mp.mpf(q) / 4)         # Eq. (1)
        print(f"${q}$ & ${A}$ & ${B}$ & ${kap}$ & ${mp.nstr(closed, 15, strip_zeros=False)}$\\\\")
        print(f"      |(68) - (1)| = {mp.nstr(abs(closed - direct), 3)},"
              f"  |(60) - (1)| = {mp.nstr(abs(gamma_form - direct), 3)}")


if __name__ == "__main__":
    main()
