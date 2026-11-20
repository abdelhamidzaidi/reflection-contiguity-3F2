"""
pslq_search.py
==============

Integer relation experiment of Remark 10 (Section 7). We look for a relation

    c_0 F(2/3) + c_1 x_1 + ... + c_17 x_17 = 0

with integer coefficients, where x_1, ..., x_17 are the constants of
Eq. (79), in the order given there. F(2/3) is computed with f3f2.hyp_unit.

Settings (Remark 10):
  * PSLQ algorithm (Ferguson, Bailey and Arno, Math. Comp. 1999), in the
    implementation mpmath.pslq of mpmath 1.3.0, applied to the 18 unscaled
    numbers;
  * working precision 120 decimal digits, repeated at 240 digits;
  * default tolerance of mpmath.pslq, 2^(-floor(0.75 p)), p the precision in bits;
  * coefficient bound maxcoeff = 10^5 and at most maxsteps = 10^7 iterations.

mpmath.pslq stops (a) when it finds a relation, (b) when its lower bound for
the Euclidean norm of any integer relation exceeds maxcoeff, (c) when the
precision is exhausted, or (d) after maxsteps iterations. The function run
captures the verbose trace of mpmath.pslq and reports the number of
iterations, the final norm bound and the reason for termination.

A negative outcome only means that no relation with small coefficients
exists among these 18 numbers. It says nothing about other constants and
does not show that F(2/3) has no closed form.
"""

import contextlib
import io
import json
import os
import re

import mpmath as mp

import f3f2


def constants():
    """The 17 constants of Eq. (79), in the order given there.

    With g = Gamma(1/3)^3: gamma_1 = g/pi and gamma_2 = pi^3/g.
    """
    g = mp.gamma(mp.mpf(1) / 3) ** 3
    s3, l3 = mp.sqrt(3), mp.log(3)
    return [1, mp.pi * s3, l3, g / mp.pi, s3 * g / mp.pi, mp.pi ** 3 / g, s3 * mp.pi ** 3 / g,
            mp.pi ** 2, l3 ** 2, mp.pi * s3 * l3, s3 * mp.clsin(2, 2 * mp.pi / 3),
            g / mp.pi * l3, s3 * g / mp.pi * l3, mp.pi ** 3 / g * l3, s3 * mp.pi ** 3 / g * l3,
            g, s3 * mp.pi ** 2 / g]


NAMES = ["1", "pi*sqrt(3)", "log(3)", "gamma_1", "sqrt(3)*gamma_1", "gamma_2",
         "sqrt(3)*gamma_2", "pi^2", "log(3)^2", "pi*sqrt(3)*log(3)",
         "sqrt(3)*Cl_2(2*pi/3)", "gamma_1*log(3)", "sqrt(3)*gamma_1*log(3)",
         "gamma_2*log(3)", "sqrt(3)*gamma_2*log(3)", "Gamma(1/3)^3",
         "sqrt(3)*pi^2/Gamma(1/3)^3"]


def write_json(dps, results):
    """Machine-readable basis (names and values) and settings/outcomes."""
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(out, exist_ok=True)
    f3f2.set_precision(dps)
    values = [f3f2.F(mp.mpf(2) / 3)] + constants()
    basis = {"definitions": {"gamma_1": "Gamma(1/3)^3/pi", "gamma_2": "pi^3/Gamma(1/3)^3",
                             "Cl_2": "Clausen function"},
             "vector": [{"name": "F(2/3)", "value": mp.nstr(values[0], dps)}] +
                       [{"name": n, "value": mp.nstr(v, dps)} for n, v in zip(NAMES, values[1:])]}
    with open(os.path.join(out, "pslq_basis.json"), "w", encoding="utf-8") as fh:
        json.dump(basis, fh, indent=1)
    with open(os.path.join(out, "pslq_result.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)


def run(dps, maxcoeff=10 ** 5, maxsteps=10 ** 7):
    """One PSLQ run at dps decimal digits; returns (relation, iterations, norm bound, reason)."""
    f3f2.set_precision(dps)
    basis = [f3f2.F(mp.mpf(2) / 3)] + constants()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rel = mp.pslq(basis, maxcoeff=maxcoeff, maxsteps=maxsteps, verbose=True)
    trace = buf.getvalue()
    steps = re.findall(r"^(\d+)/\d+:", trace, flags=re.M)
    norms = re.findall(r"Norm: (\S+)", trace)
    tol = re.search(r"tol (\S+)", trace)
    if rel is not None:
        reason = "relation found"
    elif norms and norms[-1] != "inf" and int(norms[-1]) >= maxcoeff:
        reason = "lower bound for the norm of a relation exceeded maxcoeff"
    elif steps and int(steps[-1]) >= maxsteps:
        reason = "iteration limit reached"
    else:
        reason = "precision exhausted"
    return rel, (int(steps[-1]) if steps else 0), (norms[-1] if norms else None), \
        (tol.group(1) if tol else None), reason


def main():
    results = []
    for dps in (120, 240):
        rel, it, norm, tol, reason = run(dps)
        results.append({"working_precision_digits": dps, "implementation": f"mpmath {mp.__version__} pslq",
                         "scaling": "none", "tolerance": tol, "maxcoeff": 10 ** 5, "maxsteps": 10 ** 7,
                         "iterations": it, "final_norm_bound": norm, "termination": reason,
                         "relation": rel})
        print(f"PSLQ at {dps} digits, 17 constants, maxcoeff = 10^5: relation = {rel}")
        print(f"    tolerance = {tol}, iterations = {it}, final norm bound = {norm}")
        print(f"    termination: {reason}")
    write_json(120, results)
    return results


if __name__ == "__main__":
    main()
