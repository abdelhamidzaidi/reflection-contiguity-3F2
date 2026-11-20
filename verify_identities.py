"""
verify_identities.py
====================

Numerical verification of the identities of the manuscript that are not
already covered by the table scripts. Each block prints the absolute value
of the difference between the two sides. The checks are those reported in
Section 11, together with the special values quoted in the text:

  [0] the fast summation routine f3f2.hyp_unit against mpmath's general
      routine hyp3f2(..., 1) (function f3f2.F_mpmath), for F(alpha) at five
      values of alpha; this is an independent check of the evaluator used in
      all other scripts;
  [1] Theorem 1, Eq. (16): U(beta) + U(1-beta) = D(beta) for
      alpha in {0.7, 2, 2.35} and beta in {0.3, 1.6, -0.45};
  [2] Eq. (27): the case beta = 1/2 (a special case of Dixon's theorem);
  [3] Lemma 3, Eq. (29) and Corollary 2: contiguous relation and the two
      exact values U(1+alpha) = 1/(2alpha), U(-alpha) = -1/(2alpha);
  [4] Lemma 4(i), Eq. (40): Lambda(beta) + Lambda(1-beta) identity;
  [5] Theorem 4, Eq. (61), q = 3, 5, ..., 13, against the series (1),
      at a working precision of 50 digits, and the symmetry t_{q-2-j} = t_j
      of Remark 6 in exact rational arithmetic;
  [6] Proposition 2, Eq. (74), m in {2, 4, 5, 7, 8, 10}, and Eq. (78);
  [7] Remark 1, Eq. (28): F(7/2) from Theorem 3.3 of the AIMS 2025 paper,
      and the incorrect value of its Corollary 3.2 for comparison;
  [8] Lemma 2(i) at integer alpha (terminating case);
  [9] Theorem 2: the homogeneous solution W of the proof satisfies the
      homogeneous relations (36), (37), and W(2alpha) = Gamma(alpha)/Gamma(3alpha);
 [10] Proposition 3, Eqs. (80)-(82): integral and Beta representations;
 [11] Eqs. (90) and (96): the fractional integral of I_3^{3/4} over [0, 2],
      computed from Eq. (2) by direct quadrature (20 digits, slow).
"""

import mpmath as mp

import f3f2

f3f2.set_precision(40)


def show(label, value):
    print(f"  {label:<55s} {mp.nstr(value, 3)}")


def main(run_quadrature=True):
    print("[0] hyp_unit versus mpmath hyp3f2, F(alpha), 40 digits")
    for a in (mp.mpf("0.55"), mp.mpf(2) / 3, mp.mpf("1.3"), mp.mpf("2.6"), mp.mpf("4.25")):
        show(f"alpha={mp.nstr(a, 4)}", abs(f3f2.F(a) - f3f2.F_mpmath(a)))
    # speed comparison: median of 5 timings of each routine (machine dependent)
    import statistics
    import time
    ratios = []
    for a in (mp.mpf("0.55"), mp.mpf(2) / 3, mp.mpf("1.3"), mp.mpf("2.6"), mp.mpf("4.25")):
        tf, tm = [], []
        for _ in range(5):
            t0 = time.perf_counter(); f3f2.F(a); tf.append(time.perf_counter() - t0)
            t0 = time.perf_counter(); f3f2.F_mpmath(a); tm.append(time.perf_counter() - t0)
        ratios.append(statistics.median(tm) / statistics.median(tf))
    print("  speed ratio mpmath/hyp_unit (median of 5 runs, machine dependent): "
          + ", ".join(f"{r:.0f}" for r in ratios))

    print("[1] Theorem 1, Eq. (16)")
    for a in (mp.mpf("0.7"), mp.mpf(2), mp.mpf("2.35")):
        for b in (mp.mpf("0.3"), mp.mpf("1.6"), mp.mpf("-0.45")):
            show(f"alpha={mp.nstr(a, 3)}, beta={mp.nstr(b, 3)}", f3f2.reflection_residual(a, b))

    print("[2] Eq. (27), Dixon special case")
    for a in (mp.mpf("0.8"), mp.mpf("1.9")):
        lhs = f3f2.hyp_unit([1 - a, 1, mp.mpf(1) / 2], [a + 1, mp.mpf(3) / 2])
        rhs = mp.pi * a * mp.gamma(a) ** 2 / (4 * mp.gamma(a + mp.mpf(1) / 2) ** 2)
        show(f"alpha={mp.nstr(a, 3)}", abs(lhs - rhs))

    print("[3] Lemma 3, Eq. (29), and Corollary 2")
    for a, b in ((mp.mpf("0.7"), mp.mpf("1.9")), (mp.mpf("2.3"), mp.mpf("-0.4"))):
        show(f"contiguous residual alpha={mp.nstr(a, 3)}, beta={mp.nstr(b, 3)}",
             f3f2.contiguous_residual(a, b))
    a = mp.mpf("0.7")
    show("U(1+alpha) - 1/(2alpha), alpha=0.7", abs(f3f2.U(a, 1 + a) - 1 / (2 * a)))
    show("U(-alpha) + 1/(2alpha), alpha=0.7", abs(f3f2.U(a, -a) + 1 / (2 * a)))

    print("[4] Lemma 4(i), Eq. (40)")
    for a, b in ((mp.mpf("0.7"), mp.mpf("0.45")), (mp.mpf("1.9"), mp.mpf("2.45"))):
        lhs = f3f2.Lam(a, b) + f3f2.Lam(a, 1 - b)
        rhs = 2 * mp.sin(mp.pi * b) * mp.cos(mp.pi * a) / mp.sin(mp.pi * (b - a)) * f3f2.Lam(a, b)
        show(f"alpha={mp.nstr(a, 3)}, beta={mp.nstr(b, 3)}", abs(lhs - rhs))

    print("[5] Theorem 4, Eq. (61), 50 digits")
    with mp.workdps(50):
        for q in range(3, 14, 2):
            show(f"q={q}", abs(f3f2.F_quarter_AB(q) - f3f2.F(mp.mpf(q) / 4)))
    from fractions import Fraction
    for q in range(3, 14, 2):
        a = Fraction(q, 4)
        t = []
        for j in range(q - 1):
            num = Fraction(1)
            for i in range(1, j + 1):
                num *= a - i
            den = 2 * a - 1 - j
            for i in range(1, j + 2):
                den *= 3 * a - i
            t.append(num / den)
        print(f"  q={q}: terms t_j = {[str(x) for x in t]}, palindromic: {t == t[::-1]}")

    print("[6] Proposition 2, Eq. (74), and Eq. (78)")
    for m in (2, 4, 5, 7, 8, 10):
        show(f"m={m}", abs(f3f2.F_thirds(m) - f3f2.F(mp.mpf(m) / 3)))
    third = mp.mpf(1) / 3
    rhs57 = mp.mpf(8) / 3 - mp.mpf(4) / 3 * f3f2.hyp_unit([third, third, 1], [4 * third, 5 * third])
    show("Eq. (78)", abs(f3f2.F(2 * third) - rhs57))

    print("[7] Remark 1, Eq. (28)")
    rat, c = f3f2.F_half_integer_exact(3)
    print(f"  Theorem 3.3 of AIMS 2025 at p=3: {rat} + ({c}) log(2)")
    show("|Eq. (28) - series (1)|", abs(f3f2.F_half_integer(3) - f3f2.F(mp.mpf(7) / 2)))
    wrong = mp.mpf(221158) / 415701 - mp.mpf(70) / 138567 * mp.log(2)
    show("|Corollary 3.2 of AIMS 2025 - series (1)|", abs(wrong - f3f2.F(mp.mpf(7) / 2)))
    for p in (1, 2, 4):
        show(f"Theorem 3.3 of AIMS 2025, p={p}", abs(f3f2.F_half_integer(p) - f3f2.F(p + mp.mpf(1) / 2)))

    print("[8] Integer alpha (terminating series)")
    for n in (1, 2, 3, 4):
        val = f3f2.F_integer_exact(n)
        show(f"F({n}) = {val}", abs(mp.mpf(val.numerator) / val.denominator - f3f2.F(n)))

    print("[9] Theorem 2, homogeneous solution (alpha = 0.7)")
    a = mp.mpf("0.7")
    for j in (-2, 0, 3):
        b = 2 * a + j
        hom = ((a + b - 1) * f3f2.W_homogeneous(a, b)
               - (b - a - 1) * f3f2.W_homogeneous(a, b - 1))
        show(f"homogeneous (36) on C_1, beta=2alpha{j:+d}", abs(hom))
        bb = 1 - 2 * a + j
        hom2 = ((a + bb - 1) * f3f2.W_homogeneous(a, bb)
                - (bb - a - 1) * f3f2.W_homogeneous(a, bb - 1))
        show(f"homogeneous (36) on C_2, beta=1-2alpha{j:+d}", abs(hom2))
        show(f"homogeneous (37), beta=2alpha{j:+d}",
             abs(f3f2.W_homogeneous(a, b) + f3f2.W_homogeneous(a, 1 - b)))
    show("W(2alpha) - Gamma(alpha)/Gamma(3alpha)",
         abs(f3f2.W_homogeneous(a, 2 * a) - mp.gamma(a) / mp.gamma(3 * a)))

    print("[10] Proposition 3, Eqs. (80)-(82), alpha = 0.8")
    a = mp.mpf("0.8")
    Fa = f3f2.F(a)
    i58 = 2 * a * mp.quad(lambda x: x ** (2 * a - 1) * (1 - x) ** (2 * a - 1)
                          * mp.hyp2f1(2 * a, a, a + 1, x), [0, 1])
    b60 = 2 * a * mp.beta(2 * a, 2 * a) * f3f2.hyp_unit([a, 2 * a, 2 * a], [a + 1, 4 * a])
    show("Eq. (80)", abs(i58 - Fa))
    show("Eq. (82)", abs(b60 - Fa))
    with mp.workdps(20):
        i59 = 2 * a ** 2 * mp.quad(lambda x, y: x ** (2 * a - 1) * (1 - x) ** (2 * a - 1)
                                   * y ** (a - 1) * (1 - x * y) ** (-2 * a), [0, 1], [0, 1])
        show("Eq. (81), 20 digits", abs(i59 - Fa))

    if run_quadrature:
        print("[11] Eqs. (90) and (96), tau = 0, eta = 2 (20 digits)")
        with mp.workdps(20):
            direct = f3f2.I3_integral_direct(mp.mpf(3) / 4, 0, 2)
            closed = (mp.mpf(8) / 5 * mp.mpf(2) ** (mp.mpf(5) / 4)
                      * (1 - 2 * mp.pi ** 3 / mp.gamma(mp.mpf(1) / 4) ** 4))
            show("Eq. (96) versus Eq. (2)", abs(direct - closed))
            show("Eq. (90) versus Eq. (2)", abs(direct - f3f2.I3_integral_closed(mp.mpf(3) / 4, 0, 2)))


if __name__ == "__main__":
    main()
