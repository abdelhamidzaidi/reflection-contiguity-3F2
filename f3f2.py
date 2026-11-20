"""
f3f2.py
=======

Shared definitions for the supplementary scripts of the article

    A. T. Zaidi, "Reflection, Contiguity and a Uniqueness Classification for a
    Functional System Satisfied by the Hypergeometric Family
    3F2(1-alpha,1,2alpha;alpha+1,2alpha+1;1)".

Every function below implements one formula of the manuscript. Equation,
lemma, theorem and table numbers in the comments refer to the manuscript.
The notation follows Table 1 of the manuscript:

    alpha            real parameter, alpha > 1/2
    beta             auxiliary parameter
    r_k(alpha)       (1-alpha)_k / (alpha+1)_k                       Eq. (11)
    U_alpha(beta)    sum_{k>=0} r_k(alpha) / (k + beta)               Eq. (12)
    F(alpha)         3F2(1-alpha, 1, 2alpha; alpha+1, 2alpha+1; 1)   Eq. (1)
    G(alpha)         3F2(1-alpha, 1, 1-2alpha; alpha+1, 2-2alpha; 1) Eq. (24)
    D_alpha(beta)    right-hand side of the reflection formula        Eq. (15)
    R(alpha)         D_alpha(2 alpha)                                 Eq. (25)
    Lambda, lambda   Gamma ratios                                     Eq. (33)
    E_alpha, Omega_alpha  entire functions of Lemma 4                  Eq. (39)
    S_q, A_q, B_q    quantities of Section 6                          Eqs. (59)-(61), (68)
    kappa_q          Gamma(1/4)^4/pi or pi^3/Gamma(1/4)^4             Eq. (68)

All floating-point work uses mpmath (The mpmath Development Team, version 1.3.0);
exact rational work uses Python's fractions.Fraction.

Series at unit argument are summed by the function hyp_unit below, which
combines an exact partial sum with an asymptotic expansion of the tail
(Stirling's series and the Hurwitz Zeta function). It is 20 to 50 times
faster than mpmath's general routine hyp3f2(..., 1) and reaches the full
working precision. mpmath's routine is kept as F_mpmath for an independent
cross-check (script verify_identities.py).

Author: Abdelhamid Taieb Zaidi, Department of Mathematics, College of Science,
Qassim University, Saudi Arabia.
"""

import math
from fractions import Fraction
from functools import lru_cache
from math import factorial

import mpmath as mp

# Default working precision (decimal digits). Individual scripts may raise it.
mp.mp.dps = 50


def set_precision(dps):
    """Set the mpmath working precision to `dps` decimal digits."""
    mp.mp.dps = dps


# ---------------------------------------------------------------------------
# Fast summation of Hypergeometric series at unit argument
# ---------------------------------------------------------------------------
#
# Let  S = sum_{k>=0} t_k,  t_k = (a_1)_k ... (a_p)_k / ((b_1)_k ... (b_{p-1})_k k!),
# with excess s = sum(b) - sum(a) > 0, so that S = pF_{p-1}(a; b; 1).
# We write S = sum_{k<N} t_k + sum_{k>=N} t_k.
#
# 1. The partial sum is computed from the term ratio
#        t_{k+1}/t_k = prod(a_i + k) / (prod(b_j + k) (k + 1)).
# 2. For the tail we use t_k = K prod Gamma(k+a_i) / (prod Gamma(k+b_j) Gamma(k+1)),
#        K = prod Gamma(b_j) / prod Gamma(a_i),
#    and Stirling's series (DLMF 5.11.8)
#        log Gamma(x+h) ~ (x+h-1/2) log(x) - x + log(2 pi)/2
#                         + sum_{n>=1} (-1)^{n+1} B_{n+1}(h) / (n(n+1) x^n),
#    with x = k + c and h = a_i - c, b_j - c or 1 - c, where c is the mean of
#    all parameters (this keeps the arguments h of the Bernoulli polynomials
#    B_{n+1} small). All terms except the power x^{-(s+1)} and the series in
#    1/x cancel, so that
#        t_k ~ K x^{-(s+1)} exp(sum_n c_n x^{-n}) = K sum_j e_j x^{-(s+1+j)},
#        c_n = (-1)^{n+1} [sum B_{n+1}(a_i-c) - sum B_{n+1}(b_j-c) - B_{n+1}(1-c)] / (n(n+1)),
#    where e_j follows from the recurrence j e_j = sum_{n=1}^{j} n c_n e_{j-n}, e_0 = 1.
# 3. Summing over k >= N gives the tail
#        K sum_{j=0}^{J} e_j zeta(s+1+j, N+c),
#    with the Hurwitz Zeta function zeta(sigma, X) = sum_{k>=0} (k+X)^{-sigma}.
#
# The expansion is asymptotic. Its j-th term behaves like
# (j/(2 pi e X))^j, X = N + c, so a large starting index N (cheap, since the
# partial sum costs little) allows a small number J of Zeta evaluations.
# The function _plan chooses N and J for the requested precision, and adds
# guard digits for the cancellation that occurs in the Bernoulli polynomials
# when the parameters are spread out.


@lru_cache(maxsize=None)
def _bernoulli_numbers(M, prec):
    """Bernoulli numbers B_0, ..., B_M at binary precision `prec` (cached)."""
    with mp.workprec(prec):
        return tuple(mp.bernoulli(k) for k in range(M + 1))


def _bernoulli_polynomials(h, M, B):
    """[B_0(h), ..., B_M(h)] from B_m(h) = sum_k C(m,k) B_k h^(m-k)."""
    powers = [mp.mpf(1)]
    for _ in range(M):
        powers.append(powers[-1] * h)
    values = []
    for m in range(M + 1):
        total, binom = mp.mpf(0), 1
        for k in range(m + 1):
            if B[k]:
                total += binom * B[k] * powers[m - k]
            binom = binom * (m - k) // (k + 1)
        values.append(total)
    return values


def _plan(dps, spread, c):
    """Choose the start N of the tail, the number J of tail terms and the
    number of guard digits, for a target of `dps` correct digits.

    J is the smallest integer with J log10(X/r) >= dps + 6, where X = N + c
    and r = max(J/(2 pi e), spread + 1) bounds the growth of e_j.
    """
    N = 2 * dps + int(10 * spread) + 20
    X = N + c
    J = 2
    while J < N and J * math.log10(X / max(J / (2 * math.pi * math.e), spread + 1)) < dps + 6:
        J += 1
    guard = 15 + int(0.5 * (J + 1) * math.log10(1 + spread))
    return N, J, guard


def hyp_unit(a, b):
    """pF_{p-1}(a_1, ..., a_p; b_1, ..., b_{p-1}; 1), len(a) = len(b) + 1,
    with excess sum(b) - sum(a) > 0, at the current working precision.

    Terminating series (some a_i in {0, -1, -2, ...}) are summed exactly up to
    their last term. Otherwise the method described above is used.
    """
    dps = mp.mp.dps
    am = [mp.mpf(x) for x in a]
    bm = [mp.mpf(x) for x in b]
    for x in am:
        if x <= 0 and x == mp.floor(x):                  # terminating series
            with mp.workdps(dps + 10):
                t, total = mp.mpf(1), mp.mpf(1)
                for k in range(int(-x)):
                    t *= mp.fprod(ai + k for ai in am) / (mp.fprod(bj + k for bj in bm) * (k + 1))
                    total += t
            return +total
    params = am + bm + [mp.mpf(1)]
    c_float = float(mp.fsum(params) / len(params))
    spread = float(max(abs(x - c_float) for x in params))
    N, J, guard = _plan(dps, spread, c_float)
    with mp.workdps(dps + guard):
        a_ = [mp.mpf(x) for x in a]
        b_ = [mp.mpf(x) for x in b]
        c = mp.fsum(a_ + b_ + [mp.mpf(1)]) / len(params)
        excess = mp.fsum(b_) - mp.fsum(a_)
        # 1. partial sum t_0 + ... + t_{N-1}
        t, partial = mp.mpf(1), mp.mpf(0)
        for k in range(N):
            partial += t
            t *= mp.fprod(ai + k for ai in a_) / (mp.fprod(bj + k for bj in b_) * (k + 1))
        # 2. coefficients c_n and e_j of the asymptotic expansion
        K = mp.fprod(mp.gamma(bj) for bj in b_) * mp.fprod(mp.rgamma(ai) for ai in a_)
        B = _bernoulli_numbers(J + 1, mp.mp.prec)
        bsum = [mp.mpf(0)] * (J + 2)
        for h, sign in [(x - c, 1) for x in a_] + [(x - c, -1) for x in b_] + [(1 - c, -1)]:
            for m, value in enumerate(_bernoulli_polynomials(h, J + 1, B)):
                bsum[m] += sign * value
        cn = [mp.mpf(0)] + [(-1) ** (n + 1) * bsum[n + 1] / (n * (n + 1)) for n in range(1, J + 1)]
        e = [mp.mpf(1)] + [mp.mpf(0)] * J
        for j in range(1, J + 1):
            e[j] = mp.fsum(n * cn[n] * e[j - n] for n in range(1, j + 1)) / j
        # 3. tail through the Hurwitz Zeta function
        X = N + c
        tail = mp.fsum(e[j] * mp.zeta(excess + 1 + j, X) for j in range(J + 1))
        return +(partial + K * tail)


# ---------------------------------------------------------------------------
# Section 2: the series F, G and U
# ---------------------------------------------------------------------------

def F(alpha):
    """F(alpha) = 3F2(1-alpha, 1, 2alpha; alpha+1, 2alpha+1; 1), Eq. (1).

    The series converges because its parametric excess equals 2*alpha > 0.
    """
    a = mp.mpf(alpha)
    return hyp_unit([1 - a, 1, 2 * a], [a + 1, 2 * a + 1])


def F_mpmath(alpha):
    """F(alpha) computed with mpmath's general routine hyp3f2 (slower);
    used only as an independent check of hyp_unit."""
    a = mp.mpf(alpha)
    return mp.hyp3f2(1 - a, 1, 2 * a, a + 1, 2 * a + 1, 1)


def G(alpha):
    """Companion series G(alpha), Eq. (24); requires 2*alpha not an integer."""
    a = mp.mpf(alpha)
    return hyp_unit([1 - a, 1, 1 - 2 * a], [a + 1, 2 - 2 * a])


def U(alpha, beta):
    """Auxiliary series U_alpha(beta), Eq. (12).

    For beta not in {0, -1, -2, ...} one has
        beta * U_alpha(beta) = 3F2(1-alpha, 1, beta; alpha+1, beta+1; 1),
    see the remark after Eq. (12). This identity is used for the evaluation.
    """
    a, b = mp.mpf(alpha), mp.mpf(beta)
    return hyp_unit([1 - a, 1, b], [a + 1, b + 1]) / b


def U_partial(alpha, beta, N):
    """Partial sum sum_{k=0}^{N-1} r_k/(k+beta) of Eq. (12), computed directly.

    Used only as an independent check of U(alpha, beta) for moderate N.
    """
    a, b = mp.mpf(alpha), mp.mpf(beta)
    r, s = mp.mpf(1), mp.mpf(0)
    for k in range(N):
        s += r / (k + b)
        r *= (k + 1 - a) / (k + 1 + a)      # r_{k+1} = r_k (k+1-alpha)/(k+1+alpha)
    return s


# ---------------------------------------------------------------------------
# Section 3: the reflection formula (Theorem 1, Corollary 1)
# ---------------------------------------------------------------------------

def D(alpha, beta):
    """D_alpha(beta) of Eq. (15): pi*alpha*Gamma(alpha)^2 /
    (sin(pi beta) Gamma(alpha+beta) Gamma(1+alpha-beta)), beta not an integer.

    mpmath's rgamma(z) = 1/Gamma(z) vanishes at the poles of Gamma, which is
    the convention stated after Eq. (15).
    """
    a, b = mp.mpf(alpha), mp.mpf(beta)
    return (mp.pi * a * mp.gamma(a) ** 2 * mp.rgamma(a + b) * mp.rgamma(1 + a - b)
            / mp.sin(mp.pi * b))


def R(alpha):
    """R(alpha) = alpha Gamma(alpha)^3 / (2 cos(pi alpha) Gamma(3 alpha)), Eq. (25)."""
    a = mp.mpf(alpha)
    return a * mp.gamma(a) ** 3 / (2 * mp.cos(mp.pi * a) * mp.gamma(3 * a))


def reflection_residual(alpha, beta):
    """|U_alpha(beta) + U_alpha(1-beta) - D_alpha(beta)|, Theorem 1, Eq. (16)."""
    return abs(U(alpha, beta) + U(alpha, 1 - beta) - D(alpha, beta))


def corollary1_residual(alpha):
    """|F/(2alpha) + G/(1-2alpha) - R|, Corollary 1, Eq. (25)."""
    a = mp.mpf(alpha)
    return abs(F(a) / (2 * a) + G(a) / (1 - 2 * a) - R(a))


# ---------------------------------------------------------------------------
# Section 4: contiguous relation and Gamma ratios
# ---------------------------------------------------------------------------

def contiguous_residual(alpha, beta):
    """Residual of Lemma 3, Eq. (29):
    (alpha+beta-1) U(beta) - (beta-alpha-1) U(beta-1) - alpha/(beta-1)."""
    a, b = mp.mpf(alpha), mp.mpf(beta)
    return abs((a + b - 1) * U(a, b) - (b - a - 1) * U(a, b - 1) - a / (b - 1))


def Lam(alpha, beta):
    """Lambda(beta) = Gamma(beta+alpha)/Gamma(beta-alpha), Eq. (33)."""
    a, b = mp.mpf(alpha), mp.mpf(beta)
    return mp.gamma(b + a) * mp.rgamma(b - a)


def lam(alpha, beta):
    """lambda(beta) = Gamma(beta+alpha-1)/Gamma(beta-alpha), Eq. (33)."""
    a, b = mp.mpf(alpha), mp.mpf(beta)
    return mp.gamma(b + a - 1) * mp.rgamma(b - a)


# ---------------------------------------------------------------------------
# Section 5: homogeneous solutions used in the proof of Theorem 2
# ---------------------------------------------------------------------------

def in_first_class(alpha, beta, tol=1e-30):
    """True if beta lies in C_1 = 2alpha + Z (first part of Eq. (35))."""
    x = mp.mpf(beta) - 2 * mp.mpf(alpha)
    return abs(x - mp.nint(x)) < tol


def W_homogeneous(alpha, beta):
    """Normalized homogeneous solution W_1 of the system (36)-(38), Theorem 2(ii),
    case 4alpha not in Z and 3alpha not in Z.

    W_1 = Omega_alpha / (cos(2 pi alpha) sin(pi alpha)) on C_alpha, that is,
    on C_1 = 2alpha + Z:  W_1(beta) = 1/Lambda(beta);
    on C_2 = 1 - C_1:     W_1(beta) = -W_1(1 - beta) = -1/Lambda(1 - beta).
    """
    if in_first_class(alpha, beta):
        return 1 / Lam(alpha, beta)
    return -1 / Lam(alpha, 1 - mp.mpf(beta))


def E_entire(alpha, beta):
    """E_alpha(beta) = 1/(Gamma(alpha + beta) Gamma(1 + alpha - beta)), Eq. (39).

    Entire in beta (mpmath's rgamma is the entire function 1/Gamma); beta may
    be complex.
    """
    return mp.rgamma(alpha + beta) * mp.rgamma(1 + alpha - beta)


def Omega(alpha, beta):
    """Omega_alpha(beta) = pi cos(pi beta) E_alpha(beta), Eq. (39).

    Entire homogeneous solution of the contiguous relation and of the
    reflection formula on the whole complex plane (Lemma 4(iii)).
    """
    return mp.pi * mp.cos(mp.pi * beta) * E_entire(alpha, beta)


def Omega_at_2alpha(alpha):
    """Closed form of Omega_alpha(2 alpha), Lemma 4(v)."""
    return mp.cos(2 * mp.pi * alpha) * mp.sin(mp.pi * alpha) * mp.gamma(alpha) / mp.gamma(3 * alpha)


# ---------------------------------------------------------------------------
# Section 6: quarter-integer lattice (Theorem 4, Corollary 3, Table 3)
# ---------------------------------------------------------------------------

def S_q(q):
    """Finite sum S_q of Eq. (59), alpha = q/4 with q odd, q >= 3."""
    a = mp.mpf(q) / 4
    return mp.fsum(mp.gamma(3 * a - 1 - j) / ((2 * a - 1 - j) * mp.gamma(a - j))
                   for j in range(q - 1))


def F_quarter_gamma(q):
    """Right-hand side of Eq. (60), Theorem 4."""
    a = mp.mpf(q) / 4
    return (a ** 2 * mp.gamma(a) / mp.gamma(3 * a)
            * (mp.gamma(a) ** 2 * mp.sin(3 * mp.pi * a) / mp.sin(2 * mp.pi * a) + S_q(q)))


def A_q_exact(q):
    """Rational constant A_q of Corollary 3: the finite sum in Eq. (61),
        A_q = alpha^2 sum_{j=0}^{q-2} prod_{i=1}^{j}(alpha-i)
              / ((2alpha-1-j) prod_{i=1}^{j+1}(3alpha-i)),
    computed in exact rational arithmetic."""
    a = Fraction(q, 4)
    total = Fraction(0)
    for j in range(q - 1):
        num = Fraction(1)
        for i in range(1, j + 1):
            num *= a - i
        den = 2 * a - 1 - j
        for i in range(1, j + 2):
            den *= 3 * a - i
        total += num / den
    return a * a * total


def _gamma_reduction(x):
    """Write Gamma(x) = c * Gamma(x0) with x0 in {1/4, 3/4} and rational c,
    using Gamma(z+1) = z Gamma(z). Returns (c, x0) with c a Fraction."""
    x = Fraction(x)
    x0 = x - (x.numerator // x.denominator)          # fractional part
    c = Fraction(1)
    y = x0
    while y < x:                                      # shift upwards
        c *= y
        y += 1
    while y > x:                                      # shift downwards
        y -= 1
        c /= y
    return c, x0


def kappa(q):
    """kappa_q of Eq. (68): Gamma(1/4)^4/pi if q = 1 (mod 4), pi^3/Gamma(1/4)^4 if q = 3 (mod 4)."""
    g4 = mp.gamma(mp.mpf(1) / 4) ** 4
    return g4 / mp.pi if q % 4 == 1 else mp.pi ** 3 / g4


def B_q_exact(q):
    """Rational constant B_q of Corollary 3, Eq. (68), following its proof.

    Gamma(alpha)^3/Gamma(3alpha) is reduced with _gamma_reduction and the
    identity Gamma(3/4) = pi sqrt(2)/Gamma(1/4); the factor sqrt(2) cancels
    against sin(3 pi alpha) = +-sqrt(2)/2, while sin(2 pi alpha) = +-1.
    """
    a = Fraction(q, 4)
    c1, base1 = _gamma_reduction(a)          # Gamma(alpha)  = c1 Gamma(base1)
    c2, base2 = _gamma_reduction(3 * a)      # Gamma(3alpha) = c2 Gamma(base2)
    # signs of sin(3 pi alpha)/(sqrt(2)/2) and of sin(2 pi alpha)
    s3 = int(mp.sign(mp.sin(3 * mp.pi * mp.mpf(q) / 4)))
    s2 = int(mp.sign(mp.sin(2 * mp.pi * mp.mpf(q) / 4)))
    sigma = s3 * s2
    if q % 4 == 1:
        # Gamma(a)^3/Gamma(3a) = c1^3/c2 * Gamma(1/4)^4 / (sqrt(2) pi)
        return a * a * c1 ** 3 / c2 * Fraction(sigma, 2)
    # q = 3 mod 4: Gamma(a)^3/Gamma(3a) = c1^3/c2 * 2 sqrt(2) pi^3 / Gamma(1/4)^4
    return a * a * c1 ** 3 / c2 * 2 * sigma


def F_quarter_AB(q):
    """F(q/4) = A_q + B_q kappa_q, Eq. (68)."""
    A, B = A_q_exact(q), B_q_exact(q)
    return mp.mpf(A.numerator) / A.denominator + mp.mpf(B.numerator) / B.denominator * kappa(q)


# ---------------------------------------------------------------------------
# Half-integers and integers (Remark 1)
# ---------------------------------------------------------------------------

def F_half_integer_exact(p):
    """F(p + 1/2) = rational + c * log(2), from Theorem 3.3 of [Zaidi and
    Almuthaybiri, AIMS Math. 10 (2025) 5731-5761].

    Notation of that paper, Eq. (3.5):
      K_p     = prod_{j=0}^{2p} (1 - 2p + 2j)
      a_{2p+1} = -2^{2p} (2p)! (3p)! / ((6p+1)! p!)
      E_p     = sum_{k=1}^{2p} 1/(2k)
      B_j     = sum_{k=0}^{j-1} 1/(2k - 2p + 1)
      a_j     = 2 / ((-1)^j 2^{2p} j! (2p-j)! (6p - 2j + 1))
      F = -(2p+1) K_p [ 2 a_{2p+1} log(2) + 2 a_{2p+1}(E_p + B_p) + sum_j a_j B_j ].
    Returns the pair (rational part, coefficient of log(2)) as Fractions.
    For p = 3 this gives the corrected value of Eq. (28).
    """
    K = 1
    for j in range(2 * p + 1):
        K *= 1 - 2 * p + 2 * j
    a_last = Fraction(-(2 ** (2 * p)) * factorial(2 * p) * factorial(3 * p),
                      factorial(6 * p + 1) * factorial(p))
    E = sum(Fraction(1, 2 * k) for k in range(1, 2 * p + 1))

    def Bj(j):
        return sum(Fraction(1, 2 * k - 2 * p + 1) for k in range(j))

    def aj(j):
        return Fraction(2, (-1) ** j * 2 ** (2 * p) * factorial(j)
                        * factorial(2 * p - j) * (6 * p - 2 * j + 1))

    rational = -(2 * p + 1) * K * (2 * a_last * (E + Bj(p))
                                   + sum(aj(j) * Bj(j) for j in range(1, 2 * p + 1)))
    log2_coeff = -(2 * p + 1) * K * 2 * a_last
    return rational, log2_coeff


def F_half_integer(p):
    """Floating-point value of F(p + 1/2) from F_half_integer_exact."""
    rat, c = F_half_integer_exact(p)
    return mp.mpf(rat.numerator) / rat.denominator + mp.mpf(c.numerator) / c.denominator * mp.log(2)


def F_integer_exact(n):
    """F(n) for integer n >= 1: the series (1) terminates (Remark 1).
    By Lemma 2(i), F(n) = 2n sum_{k=0}^{n-1} r_k/(k+2n), with exact r_k."""
    total, r = Fraction(0), Fraction(1)
    for k in range(n):
        total += r / (k + 2 * n)
        r *= Fraction(k + 1 - n, k + 1 + n)
    return 2 * n * total


# ---------------------------------------------------------------------------
# Section 7: companion-series identity for alpha = m/3 (Proposition 2)
# ---------------------------------------------------------------------------

def F_thirds(m):
    """Right-hand side of Eq. (74), alpha = m/3, m >= 2, 3 does not divide m.
    The only non-explicit ingredient is U_alpha(1-alpha), Eq. (75)."""
    a = mp.mpf(m) / 3
    finite = a * mp.fsum(mp.factorial(i) / ((1 - a + i) * mp.gamma(2 - 2 * a + i))
                         for i in range(m - 1))
    return (2 * a * mp.gamma(a) / mp.gamma(3 * a)
            * (mp.gamma(2 * a) * mp.sin(2 * mp.pi * a) / mp.pi * U(a, 1 - a) + finite))


# ---------------------------------------------------------------------------
# Section 9: fractional integrals
# ---------------------------------------------------------------------------

def I3_integral_closed(alpha, tau, eta):
    """Integral of I_3^alpha over [tau, eta] from Eq. (90)."""
    a = mp.mpf(alpha)
    return (mp.mpf(eta) - tau) ** (3 * a - 1) / (2 * a ** 2) * F(a)


def I3_integral_direct(alpha, tau, eta):
    """Same integral computed directly from the definition (2) by nested
    Gauss-Legendre quadrature (slow; used once, with moderate precision)."""
    a, t0, t1 = mp.mpf(alpha), mp.mpf(tau), mp.mpf(eta)

    def I3(t):
        return mp.quad(lambda s: (t - t0) ** (a - 1) * (t - s) ** (a - 1) * (t1 - s) ** (a - 1),
                       [t0, t])

    return mp.quad(I3, [t0, t1])
