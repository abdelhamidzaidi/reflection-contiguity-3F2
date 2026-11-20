"""
gosper_check.py
===============

Symbolic certificate for Lemma 5 (Section 8). The right-hand side of the
telescoping form (34) of the contiguous relation is the term

    h(beta) = alpha * lambda(beta) / (beta - 1)
            = alpha * Gamma(beta + alpha - 1) / (Gamma(beta - alpha) * (beta - 1)),

with lambda as in Eq. (33). The script

  1. computes the term ratio h(beta + 1)/h(beta) = a(beta)/b(beta) with
         a(beta) = (beta + alpha - 1)(beta - 1),   b(beta) = beta (beta - alpha);
  2. lists the zeros of a(beta) and of b(beta + j) and the values of alpha for
     which they can coincide (only alpha or 2 alpha in N*), so that (a, b, 1)
     is a Gosper form whenever 2 alpha is not an integer;
  3. computes the leading coefficient of a(beta) x(beta+1) - b(beta-1) x(beta)
     for a polynomial x of degree d, d = 0, ..., 6; it equals (2 alpha + d) x_d,
     which never vanishes for alpha > 0, so the Gosper equation (87) has no
     polynomial solution;
  4. confirms 3 by solving (87) with a generic polynomial of degree d <= 6;
  5. runs SymPy's implementation of Gosper's algorithm (Gosper, Proc. Natl.
     Acad. Sci. USA 1978) with alpha symbolic, which returns None.
"""

import sympy as sp
from sympy.concrete.gosper import gosper_term


def main():
    alpha, beta, j = sp.symbols("alpha beta j")
    h = alpha * sp.gamma(beta + alpha - 1) / (sp.gamma(beta - alpha) * (beta - 1))

    # 1. term ratio
    ratio = sp.factor(sp.gammasimp(h.subs(beta, beta + 1) / h))
    a = (beta + alpha - 1) * (beta - 1)
    b = beta * (beta - alpha)
    assert sp.simplify(ratio - a / b) == 0
    print("h(beta+1)/h(beta) =", ratio)

    # 2. Gosper form: common zeros of a(beta) and b(beta + j), j in N
    za = sp.solve(a, beta)
    zb = sp.solve(b.subs(beta, beta + j), beta)
    print("zeros of a(beta):", za, "  zeros of b(beta+j):", zb)
    for x in za:
        for y in zb:
            cond = sp.solve(sp.Eq(x, y), alpha)
            print(f"    {x} = {y}  requires alpha = {cond}")
    print("    -> a common zero needs alpha in N* or 2 alpha in N*; excluded when 2 alpha is not an integer")

    # 3. leading coefficient for deg x = d
    bm1 = sp.expand(b.subs(beta, beta - 1))
    print("a(beta) =", sp.expand(a), "   b(beta-1) =", bm1)
    for d in range(7):
        expr = sp.expand(a * (beta + 1) ** d - bm1 * beta ** d)
        lead = sp.Poly(expr, beta).LC()
        print(f"    deg x = {d}: coefficient of beta^{d + 1} is ({sp.factor(lead)}) * x_d")

    # 4. no polynomial solution of degree <= 6
    for d in range(7):
        cs = sp.symbols(f"c0:{d + 1}")
        x = sum(c * beta ** i for i, c in enumerate(cs))
        eqs = sp.Poly(sp.expand(a * x.subs(beta, beta + 1) - bm1 * x - 1), beta).coeffs()
        sol = sp.solve(eqs, cs, dict=True)
        print(f"    polynomial solutions of degree {d}: {sol}")

    # 5. SymPy's Gosper algorithm
    result = gosper_term(h, beta)
    print("SymPy gosper_term(h, beta):", result)
    print("Hypergeometric antidifference exists:", result is not None)
    return result


if __name__ == "__main__":
    main()
