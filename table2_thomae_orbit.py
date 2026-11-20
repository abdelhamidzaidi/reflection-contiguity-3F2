"""
table2_thomae_orbit.py
======================

Reproduces Table 2 of the manuscript (the Thomae orbit of F(alpha)), the
symbolic certificate used in the proof of Proposition 1, and Table B1 of
Appendix B (solution sets of the 240 condition pairs).

Method
------
1. Starting from the parameter set of Eq. (1),
       numerator (1-alpha, 1, 2alpha),  denominator (alpha+1, 2alpha+1),
   we apply the two-term Thomae transformation (57) (Bailey 1935, Section 3.2)
       3F2(a,b,c; d,e; 1) -> 3F2(d-a, e-a, s; s+b, s+c; 1),  s = d+e-a-b-c,
   to every permutation of (a,b,c) and (d,e), and repeat until no new
   parameter set appears. Parameter sets are compared up to permutations of
   the numerator and of the denominator parameters. The orbit has five
   elements, T1-T5 in Table 2: round 1 gives T2 and T3, round 2 gives T4 and
   T5, round 3 gives nothing new (function orbit_rounds).
2. For every element and every permutation of its parameters, we form the
   linear conditions in alpha that express one of the classical summation
   patterns listed in Proposition 1:
     * two disjoint pairs (a_i, b_j), (a_k, b_l) with integer differences
       (Gauss, Karlsson-Minton, Gottschalk-Maslen);
     * contiguity to Dixon:   b_1 - (1+a-b) in Z and b_2 - (1+a-c) in Z;
     * contiguity to Watson:  b_1 - (a+b+1)/2 in Z and b_2 - 2c in Z;
     * contiguity to Whipple: a + b - 1 in Z and b_1 + b_2 - 1 - 2c in Z;
     * termination (a numerator parameter in {0,-1,-2,...});
     * reduction to 2F1 (a numerator parameter equal to a denominator one).
   Each condition has the form c0 + c1*alpha in Z with rational c0, c1.
3. For each condition (or pair of conditions) we determine the complete set
   of solutions alpha, which is empty or an arithmetic progression in Q, and
   check exactly that every solution satisfies 4*alpha in Z.

The script prints the rows of Table 2 in LaTeX form, the rounds of the orbit
construction, the summary of Table B1 and the final verdict. The complete list
of the 240 condition pairs with their solution sets is written to
output/thomae_certificate.txt (function certificate).
"""

import itertools
from fractions import Fraction

import sympy as sp

al = sp.symbols("alpha")
START = ((1 - al, sp.Integer(1), 2 * al), (al + 1, 2 * al + 1))   # Eq. (1)


def canon(params):
    """Sort numerator and denominator parameters (orbit elements are sets)."""
    num = tuple(sorted((sp.expand(x) for x in params[0]), key=sp.default_sort_key))
    den = tuple(sorted((sp.expand(x) for x in params[1]), key=sp.default_sort_key))
    return num, den


def thomae_images(params):
    """All images of a parameter set under the two-term Thomae transformation."""
    out = []
    for a, b, c in itertools.permutations(params[0]):
        for d, e in itertools.permutations(params[1]):
            s = d + e - a - b - c
            out.append(canon(((d - a, e - a, s), (s + b, s + c))))
    return out


def thomae_orbit():
    """Breadth-first closure of START under the Thomae transformation."""
    seen, frontier = {canon(START)}, [canon(START)]
    while frontier:
        new = []
        for p in frontier:
            for q in thomae_images(p):
                if q not in seen:
                    seen.add(q)
                    new.append(q)
        frontier = new
    return seen


def lin(expr):
    """Return (c0, c1) as Fractions for a linear expression c0 + c1*alpha."""
    e = sp.expand(expr)
    return Fraction(str(e.coeff(al, 0))), Fraction(str(e.coeff(al, 1)))


def solution_set(conds):
    """Exact solution set of the conditions c0 + c1*alpha in Z.

    Returns None if the set is empty, 'ALL' if every real alpha works, or a
    pair (offset, period) describing {offset + k*period : k in Z}.
    Conditions with c1 = 0 are tested directly. For the others, the first
    condition gives alpha = (k - c0)/c1, and the second one is periodic in k.
    """
    active = []
    for c0, c1 in conds:
        if c1 == 0:
            if c0.denominator != 1:
                return None
        else:
            active.append((c0, c1))
    if not active:
        return "ALL"
    c0, c1 = active[0]
    if len(active) == 1:
        return (-c0 / c1, 1 / abs(c1))
    d0, d1 = active[1]
    # alpha = (k - c0)/c1 ; need d0 + d1*(k - c0)/c1 in Z ; period in k:
    t = d1 / c1
    period_k = t.denominator
    sols = [k for k in range(period_k) if (d0 + d1 * (k - c0) / c1).denominator == 1]
    if not sols:
        return None
    # all solutions: alpha = (k0 - c0)/c1 + j*period_k/c1, k0 in sols
    return [((k - c0) / c1, Fraction(period_k) / abs(c1)) for k in sols]


def all_in_quarter_lattice(sol):
    """True if every alpha in the solution set satisfies 4*alpha in Z."""
    if sol is None:
        return True
    if sol == "ALL":
        return False
    progs = sol if isinstance(sol, list) else [sol]
    return all((4 * off).denominator == 1 and (4 * per).denominator == 1 for off, per in progs)


def conditions(num, den, with_labels=False):
    """Yield (pattern name, list of linear conditions) for one orbit element.

    With with_labels=True, a third entry describes the ordering used, i.e.
    which parameters play the roles in the pattern.
    """
    fmt = lambda x: str(sp.expand(x))
    for (i, j), (k, l) in itertools.product(itertools.permutations(range(3), 2),
                                            [(0, 1), (1, 0)]):
        conds = [lin(den[k] - num[i]), lin(den[l] - num[j])]
        lab = (f"pairs (num {fmt(num[i])}, den {fmt(den[k])}), "
               f"(num {fmt(num[j])}, den {fmt(den[l])})")
        yield ("two differences", conds, lab) if with_labels else ("two differences", conds)
    for a, b, c in itertools.permutations(num):
        for d, e in itertools.permutations(den):
            lab = f"a={fmt(a)}, b={fmt(b)}, c={fmt(c)}, d={fmt(d)}, e={fmt(e)}"
            for name, conds in (("Dixon", [lin(d - (1 + a - b)), lin(e - (1 + a - c))]),
                                ("Watson", [lin(d - (a + b + 1) / 2), lin(e - 2 * c)]),
                                ("Whipple", [lin(a + b - 1), lin(d + e - 1 - 2 * c)])):
                yield (name, conds, lab) if with_labels else (name, conds)


def special_points(num, den):
    """Termination and reduction to 2F1 for alpha > 1/2.

    Termination: a numerator parameter a = c0 + c1*alpha lies in {0,-1,-2,...}.
    Reduction:   a numerator parameter equals a denominator parameter.
    Returns True if all such alpha > 1/2 satisfy 4*alpha in Z.
    """
    from math import ceil, floor
    ok = True
    half = Fraction(1, 2)
    for a in num:
        c0, c1 = lin(a)
        if c1 > 0:
            # a ranges over (c0 + c1/2, oo); integers n <= 0 in that range
            lo = c0 + c1 * half
            for n in range(floor(lo) + 1, 1):
                ok &= (4 * (n - c0) / c1).denominator == 1
        elif c1 < 0:
            # a ranges over (-oo, c0 + c1/2): infinitely many n <= 0,
            # alpha = (n - c0)/c1 forms a progression of period 1/|c1|
            ok &= all_in_quarter_lattice((-c0 / c1, 1 / abs(c1)))
    for a in num:
        for b in den:
            c0, c1 = lin(b - a)
            if c1 != 0:
                x = -c0 / c1                  # the unique alpha with b = a
                if x > half:
                    ok &= (4 * x).denominator == 1
            elif c0 == 0:
                ok = False                    # identical parameters for all alpha
    return ok


def latex_row(label, num, den):
    def fmt(x):
        return sp.latex(x).replace("\\alpha", "\\alpha")
    diffs = [[sp.expand(b - a) for a in num] for b in den]
    return (f"{label} & $({', '.join(fmt(x) for x in num)})$ & $({', '.join(fmt(x) for x in den)})$ & "
            f"$({', '.join(fmt(x) for x in diffs[0])})$ & $({', '.join(fmt(x) for x in diffs[1])})$\\\\")


def orbit_rounds():
    """Breadth-first rounds of the orbit construction (Section 5.3)."""
    names = {canon(START): "T1",
             canon(((al, 2 * al, 2 * al), (4 * al, al + 1))): "T2",
             canon(((2 * al, 2 * al, 3 * al), (4 * al, 2 * al + 1))): "T3",
             canon(((sp.Integer(1), sp.Integer(1), 3 * al), (2 * al + 1, 2 * al + 1))): "T4",
             canon(((al, 1 - al, 1 - al), (al + 1, al + 1))): "T5"}
    seen, frontier, rnd = {canon(START)}, [canon(START)], 0
    while frontier:
        rnd += 1
        new = []
        for p in frontier:
            for q in thomae_images(p):
                if q not in seen:
                    seen.add(q)
                    new.append(q)
        print(f"  round {rnd}: new sets {[names[q] for q in new] or 'none'}")
        frontier = new
    return names


def describe(sol):
    """Readable form of a solution set returned by solution_set."""
    if sol is None:
        return "empty"
    if sol == "ALL":
        return "R"
    progs = sol if isinstance(sol, list) else [sol]
    return " U ".join(sorted({f"{off % per} + {per} Z" for off, per in progs}))


def certificate(names, path):
    """Write every condition pair with its solution set; return the summary
    {pattern: {solution set: count}} reproduced in Table B1."""
    from collections import Counter, defaultdict
    summary = defaultdict(Counter)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("Certificate for Proposition 1: condition pairs c0 + c1*alpha in Z\n")
        fh.write("Orbit elements are parameter sets identified up to permutations of the\n")
        fh.write("numerator and of the denominator parameters (function canon: both lists are\n")
        fh.write("sorted after symbolic expansion). Every ordering of each set is listed.\n")
        fh.write("set | pattern | ordering | conditions | solution set\n")
        for p, name in sorted(names.items(), key=lambda kv: kv[1]):
            num, den = p
            fh.write(f"# {name}: numerator {tuple(map(str, num))}, denominator {tuple(map(str, den))}\n")
            for pattern, conds, lab in conditions(num, den, with_labels=True):
                sol = solution_set(conds)
                summary[pattern][describe(sol)] += 1
                cs = "; ".join(f"{c0} + ({c1}) alpha in Z" for c0, c1 in conds)
                fh.write(f"{name} | {pattern} | {lab} | {cs} | {describe(sol)}\n")
    return summary


def main():
    orbit = thomae_orbit()
    # order of Table 2: T1 is the set of Eq. (1); T2-T5 follow the manuscript
    order = [canon(START),
             canon(((al, 2 * al, 2 * al), (4 * al, al + 1))),
             canon(((2 * al, 2 * al, 3 * al), (4 * al, 2 * al + 1))),
             canon(((sp.Integer(1), sp.Integer(1), 3 * al), (2 * al + 1, 2 * al + 1))),
             canon(((al, 1 - al, 1 - al), (al + 1, al + 1)))]
    assert set(order) == orbit, "unexpected Thomae orbit"
    orbit = order
    print(f"Thomae orbit of F(alpha): {len(orbit)} parameter sets (Table 2)\n")
    for idx, (num, den) in enumerate(orbit, 1):
        print(latex_row(f"T{idx}", num, den))
    verdict = True
    for num, den in orbit:
        verdict &= special_points(num, den)
        for name, conds in conditions(num, den):
            if not all_in_quarter_lattice(solution_set(conds)):
                verdict = False
                print("  pattern outside (1/4)Z:", name, num, den, conds)
    print("\nOrbit construction:")
    names = orbit_rounds()
    import os
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(out, exist_ok=True)
    summary = certificate(names, os.path.join(out, "thomae_certificate.txt"))
    print("\nTable B1 (number of condition pairs per solution set):")
    for pattern, cnt in summary.items():
        print(f"  {pattern:16s} {sum(cnt.values()):3d}  " + "; ".join(f"{k} ({v})" for k, v in sorted(cnt.items())))
    print("\nProposition 1: every classical summation pattern forces 4*alpha in Z:", verdict)


if __name__ == "__main__":
    main()
