"""
verify_uniqueness.py
====================

Numerical tests of the statements of Section 5 (uniqueness for the functional
system) and Section 10 (scope of the standing assumptions). These are tests
of the formulas and of their implementation; the statements are proved in
the manuscript.

  check_omega         Lemma 4(iii): Omega_alpha satisfies the homogeneous
                      contiguous relation and the homogeneous reflection
                      relation (41) at real and complex beta; Lemma 4(v):
                      closed form of Omega_alpha(2 alpha); Lemma 4(iv): decay.
  check_piecewise     Remark 2: the piecewise forms (46)-(47) of Omega_alpha on
                      C_alpha for alpha = 2/3 and the joining relations at
                      beta = 1 - alpha and beta = 1 + alpha.
  check_growth        Theorem 3: |U_alpha| and |Omega_alpha| on the line
                      Re(beta) = 1/2; |Omega_alpha| behaves like
                      (1/4) exp(2 pi |y|) |y|^(-2 alpha).
  check_small_alpha   Proposition 4: reflection (16) and contiguous relation
                      (29) for 0 < alpha <= 1/2, F(1/4) = Gamma(1/4)^4/(32 pi)
                      (Dixon), F(1/2) = 2 log(2), the case q = 1 of (61), and
                      the fractional-integral identity (90) at alpha = 1/4.
"""

import mpmath as mp

import f3f2

POINTS = [mp.mpf("0.37"), mp.mpf("1.6"), mp.mpf("-2.3"), mp.mpc("0.2", "1.5"), mp.mpc("-1.7", "-0.8")]


def show(label, value):
    print(f"  {label:<62s} {mp.nstr(value, 3)}")


def check_omega():
    print("[1] Lemma 4: Omega_alpha(beta) = pi cos(pi beta) / (Gamma(alpha+beta) Gamma(1+alpha-beta))")
    for a in (mp.mpf("0.7"), mp.mpf(2) / 3, mp.mpf("1.3"), mp.mpf("2.35"), mp.mpc("0.8", "0.3")):
        worst_c = max(abs((a + b - 1) * f3f2.Omega(a, b) - (b - a - 1) * f3f2.Omega(a, b - 1)) for b in POINTS)
        worst_r = max(abs(f3f2.Omega(a, b) + f3f2.Omega(a, 1 - b)) for b in POINTS)
        show(f"alpha = {mp.nstr(a, 4)}: max homogeneous contiguous residual", worst_c)
        show(f"alpha = {mp.nstr(a, 4)}: max homogeneous reflection residual", worst_r)
        show(f"alpha = {mp.nstr(a, 4)}: |Omega(2alpha) - cos(2 pi a) sin(pi a) G(a)/G(3a)|",
             abs(f3f2.Omega(a, 2 * a) - f3f2.Omega_at_2alpha(a)))
    a = mp.mpf("0.7")
    for b in (mp.mpf("10.4"), mp.mpf("40.4"), mp.mpf("160.4")):
        show(f"alpha = 0.7, beta = {b}: |Omega| / (Gamma(b-a)/Gamma(b+a))",
             abs(f3f2.Omega(a, b)) / (mp.gamma(b - a) / mp.gamma(b + a)))


def check_piecewise():
    print("[2] Remark 2: alpha = 2/3, piecewise form of Omega on C_alpha and joining points")
    a = mp.mpf(2) / 3
    c = -mp.cos(mp.pi * a) * mp.sin(2 * mp.pi * a)
    worst = 0
    for n in range(-6, 8):
        b1 = -a + n                                   # C_1 = -alpha + Z
        expected = 0 if b1 <= -a + mp.mpf("1e-30") else c / f3f2.Lam(a, b1)
        worst = max(worst, abs(f3f2.Omega(a, b1) - expected))
        b2 = a + n                                    # C_2 = alpha + Z
        expected2 = 0 if b2 >= 1 + a - mp.mpf("1e-30") else -c / f3f2.Lam(a, 1 - b2)
        worst = max(worst, abs(f3f2.Omega(a, b2) - expected2))
    show("max deviation from (46)-(47) on 28 points", worst)
    show("constant -cos(pi a) sin(2 pi a) (non-zero)", c)
    b = 1 - a
    show("join at beta = 1 - alpha: (a+b-1) W(b) - (b-a-1) W(b-1)",
         (a + b - 1) * f3f2.Omega(a, b) - (b - a - 1) * f3f2.Omega(a, b - 1))
    b = 1 + a
    show("join at beta = 1 + alpha: (a+b-1) W(b) - (b-a-1) W(b-1)",
         (a + b - 1) * f3f2.Omega(a, b) - (b - a - 1) * f3f2.Omega(a, b - 1))


def check_growth():
    print("[3] Theorem 3: growth on the line Re(beta) = 1/2, alpha = 0.7")
    a = mp.mpf("0.7")
    for y in (5, 10, 20, 40):
        b = mp.mpc(mp.mpf(1) / 2, y)
        u = abs(mp.hyp3f2(1 - a, 1, b, a + 1, b + 1, 1) / b)   # complex beta: general mpmath routine
        om = abs(f3f2.Omega(a, b))
        ref = mp.exp(2 * mp.pi * y) * mp.mpf(y) ** (-2 * a) / 4
        print(f"  y = {y:3d}:  |U| = {mp.nstr(u, 6):>10s}   |Omega| / ((1/4) e^(2 pi y) y^(-2a)) = {mp.nstr(om / ref, 8)}")


def check_small_alpha():
    print("[4] Proposition 4: 0 < alpha <= 1/2")
    for a in (mp.mpf("0.3"), mp.mpf(1) / 3, mp.mpf("0.45")):
        for b in (mp.mpf("0.37"), mp.mpf("1.6"), mp.mpf("-0.45")):
            refl = abs(f3f2.U(a, b) + f3f2.U(a, 1 - b) - f3f2.D(a, b))
            cont = abs((a + b - 1) * f3f2.U(a, b) - (b - a - 1) * f3f2.U(a, b - 1) - a / (b - 1))
            show(f"alpha = {mp.nstr(a, 4)}, beta = {b}: reflection / contiguous",
                 max(refl, cont))
    q = mp.mpf(1) / 4
    closed = mp.gamma(q) ** 4 / (32 * mp.pi)
    show("|F(1/4) - Gamma(1/4)^4/(32 pi)|", abs(f3f2.F(q) - closed))
    gamma_form = q ** 2 * mp.gamma(q) ** 3 * mp.sin(3 * mp.pi * q) / (mp.gamma(3 * q) * mp.sin(2 * mp.pi * q))
    show("|Eq. (61) with q = 1 - Gamma(1/4)^4/(32 pi)|", abs(gamma_form - closed))
    show("|F(1/2) - 2 log(2)|", abs(f3f2.F(mp.mpf(1) / 2) - 2 * mp.log(2)))
    with mp.workdps(25):
        a = mp.mpf(1) / 4
        inner = lambda u: u ** (a - 1) * mp.quad(lambda v: (u - v) ** (a - 1) * (1 - v) ** (a - 1), [0, u / 2, u])
        J = mp.quad(inner, [0, mp.mpf(1) / 2, 1])
        show("alpha = 1/4: relative error of (90) by nested quadrature (tau=0, eta=1)",
             abs(J - closed / (2 * a ** 2)) / abs(J))


def main():
    f3f2.set_precision(50)
    check_omega()
    check_piecewise()
    check_growth()
    check_small_alpha()


if __name__ == "__main__":
    main()
