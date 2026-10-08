import numpy as np
import scipy as sp
from typing import Union
from decimal import Decimal, localcontext
import warnings
import os
import concurrent.futures

ArrayLike = Union[int, float, list, 'np.ndarray']

_EPS = np.finfo(float).eps
_METHODS = ('auto', 'closed_form', 'decimal', 'convolution', 'phase_type')
_CONVOLUTION_MIN_POINTS = 10 ** 5


def _as_scales(scales: ArrayLike) -> 'np.ndarray':
    """Validate `scales` and return it as a flat float array (shared by every public function)."""
    scales_array = np.atleast_1d(np.asarray(scales, dtype=float)).ravel()
    if scales_array.size == 0:
        raise ValueError("At least one scale parameter is required.")
    if not np.all(np.isfinite(scales_array)) or np.any(scales_array <= 0):
        raise ValueError("Scale parameters must be finite and strictly greater than 0.")
    return scales_array


def hypoexp_rvs(scales: ArrayLike, size: Union[int, tuple] = 1000) -> Union['np.ndarray', float]:
    """
    Generate random variables from a hypoexponential distribution.

    The hypoexponential distribution represents the sum of independent
    exponential random variables with potentially different scale parameters.

    Parameters
    ----------
    scales : array_like
        A single scale parameter or an iterable of scale parameters (must be > 0).
    size : int or tuple of ints, optional
        Defining number of random variates (default is 1000).

    Returns
    -------
    rvs : ndarray or scalar
        An array of shape `size` containing the generated random variables.

    Raises
    ------
    ValueError
        If any of the scale parameters are less than or equal to 0.
    """
    scales_array = _as_scales(scales)

    if scales_array.size == 1:
        return sp.stats.expon.rvs(scale=scales_array[0], size=size)

    size_tuple = (int(size),) if np.isscalar(size) else tuple(size)
    full_size = (scales_array.size,) + size_tuple

    # Reshape scales_array to broadcast across the size_tuple dimensions
    scale_shape = (scales_array.size,) + (1,) * len(size_tuple)
    scales_reshaped = scales_array.reshape(scale_shape)

    return np.sum(sp.stats.expon.rvs(scale=scales_reshaped, size=full_size), axis=0)


##### Evaluation methods #####
# There are several ways to compute the pdf/cdf of a hypoexponential distribution, see
# https://en.wikipedia.org/wiki/Hypoexponential_distribution. Each has strengths and weaknesses,
# so `hypoexp_pdf`/`hypoexp_cdf` try the fastest one first and fall back to slower ones only for
# the points where a numerical problem is detected. Every method below evaluates each point of `x`
# independently (except the convolution, which needs a uniform grid), so a scalar `x` or a grid
# that does not cover the whole distribution is handled exactly like any other input.


def _closed_form_method(x, scales, kind):
    r"""Closed form for distinct scales (fast, vectorized):

    pdf(x) = \sum_i w_i \omega_i^{-1} e^{-x/\omega_i},   cdf(x) = \sum_i w_i (1 - e^{-x/\omega_i}),
    where w_i = \prod_{j \neq i} \omega_i / (\omega_i - \omega_j).

    When scales are close, the denominators (\omega_i - \omega_j) make the w_i huge with alternating
    signs and the sum suffers catastrophic cancellation. We therefore also return, for each point,
    an upper bound on the relative rounding error of the float result:
        4 n eps * sum_i |term_i| / |sum_i term_i|
    (the w_i are each accurate to ~2n eps since the differences of close floats are exact, and the
    sum of n terms loses at most eps per term relative to sum|term_i|). The caller rejects points
    whose bound exceeds the tolerance and recomputes them with a slower method.

    Returns (value, relative_error_bound), both arrays of len(x). Non-finite values (overflowed
    weights) must also be treated as failures by the caller.
    """
    n = scales.size
    # Everything is vectorized over (scale i, point k). Overflow / division by zero are allowed
    # here on purpose: they produce inf/nan values that the caller treats as failed points.
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        # ratio[i, j] = omega_i / (omega_i - omega_j); the diagonal (i == j) is set to 1 so that
        # the row product gives w_i = prod_{j != i} ratio[i, j]
        ratio = scales[:, None] / (scales[:, None] - scales[None, :])
        np.fill_diagonal(ratio, 1.0)
        weights = np.prod(ratio, axis=1)
        # basis[i, k] = e^{-x_k/omega_i}/omega_i (pdf) or 1 - e^{-x_k/omega_i} (cdf; expm1 keeps
        # full relative accuracy for small x)
        if kind == 'pdf':
            basis = np.exp(-x[None, :] / scales[:, None]) / scales[:, None]
        else:
            basis = -np.expm1(-x[None, :] / scales[:, None])
        terms = weights[:, None] * basis
        value = np.sum(terms, axis=0)
        # a posteriori error bound: the n rounded terms can each be off by ~eps relative, so the
        # absolute error of the sum is ~eps * sum|terms|; dividing by |value| gives the relative
        # error, which explodes exactly when the terms cancel (close scales)
        rel_err = 4 * n * _EPS * np.sum(np.abs(terms), axis=0) / np.abs(value)
    return value, rel_err


def _decimal_terms(x_dec, scales_dec, weights, kind):
    """The n closed-form terms at one point x (Decimal), together with an upper bound on the
    magnitude each term's rounding error is proportional to (see _decimal_chunk)."""
    terms, bounds = [], []
    for w, s_i in zip(weights, scales_dec):
        e = (-x_dec / s_i).exp()
        if kind == 'pdf':
            t = w * e / s_i
            terms.append(t)
            bounds.append(abs(t))
        else:
            one_minus_e = 1 - e
            terms.append(w * one_minus_e)
            # 1 - e is computed with absolute error ~10^-prec, so its relative error can be large
            bounds.append(abs(w) * (1 + abs(one_minus_e)))
    return terms, bounds


def _decimal_weights(scales_dec):
    """Closed-form weights w_i = prod_{j != i} omega_i / (omega_i - omega_j) in the current Decimal
    context. The float scales convert exactly, so the differences are exact and the only rounding
    is in the n - 1 divisions/products per weight. O(n^2) operations; cached per precision."""
    ws = []
    for i, s_i in enumerate(scales_dec):
        w = Decimal(1)
        for j, s_j in enumerate(scales_dec):
            if i != j:
                w *= s_i / (s_i - s_j)
        ws.append(w)
    return ws


def _decimal_chunk(args):
    """Decimal closed form for a chunk of points (one argument tuple so it can be sent to a worker
    process). For each point:
      1. evaluate the n terms at precision `prec` (starting from start_digits[k]);
      2. bound the rounding error of the sum: every term went through at most ~3n + 4 roundings
         (weights, exp, products, additions), each of relative size 10^-prec, so
             |error| <= (3n + 4) * 10^-prec * sum_i bound_i;
      3. accept when the bound is below 1e-17 * |total| (the float conversion is then exact to
         double precision); otherwise raise `prec` by the number of digits the bound says are
         missing (+8 guard digits) and go back to 1. If the total is itself smaller than the
         error bound it is pure noise and cannot be used to estimate anything, so double `prec`.
    Returns (values, converged_mask); points still unconverged at max_digits are flagged False."""
    x, scales, kind, start_digits, max_digits = args
    scales_dec = [Decimal(float(s)) for s in scales]
    n = len(scales_dec)
    out = np.empty(x.size)
    ok = np.zeros(x.size, dtype=bool)
    weights_cache = {}
    with localcontext() as ctx:
        for k, x_k in enumerate(x):
            x_dec = Decimal(float(x_k))
            prec = int(start_digits[k])
            value = float('nan')
            while prec <= max_digits:
                ctx.prec = prec
                if prec not in weights_cache:
                    weights_cache[prec] = _decimal_weights(scales_dec)
                terms, bounds = _decimal_terms(x_dec, scales_dec, weights_cache[prec], kind)
                total = sum(terms, Decimal(0))
                value = float(total)
                # every term carries <= (2n + 4) roundings (weights, exp, products) and the sum
                # another n, each of relative size 10^-prec
                err = (3 * n + 4) * Decimal(10) ** (-prec) * sum(bounds, Decimal(0))
                if total != 0 and err <= Decimal('1e-17') * abs(total):
                    ok[k] = True
                    break
                if total == 0 or err > abs(total):
                    # the computed total is itself dominated by rounding error, so it cannot be
                    # used to estimate the precision needed: double instead
                    prec *= 2
                else:
                    needed = (err / (Decimal('1e-17') * abs(total))).log10()
                    prec = int(prec + float(needed)) + 8
            out[k] = value
    return out, ok


_DECIMAL_PARALLEL_MIN_COST = 0.5  # predicted seconds above which points are spread over processes


def _decimal_method(x, scales, kind, start_digits, max_digits=2000):
    """Closed form evaluated with the `decimal` module (distinct scales only).

    Arbitrary precision removes the cancellation problem of the float closed form, at the cost of
    speed (pure Python loops, cost grows with len(scales) and with the precision needed). For each
    point the precision starts at `start_digits` (chosen from the float error bound) and the result
    is accepted when a rigorous rounding-error bound computed alongside it (from sum|term_i|, the
    same quantity the float method uses) is below 1e-17 relative, which proves the float conversion
    is exact to double precision. Otherwise the precision is raised to what the bound says is
    needed and the point is re-evaluated (usually a single evaluation suffices). This replaces the
    old "integrate to 1" criterion that could never be met for a single point or a truncated grid.
    Points that still have not converged at `max_digits` are reported as failed.

    Large workloads are split over worker processes (the decimal module releases no GIL work to
    vectorize, so processes are the only way to use several cores).

    Returns (value, converged_mask).
    """
    start_digits = np.asarray(start_digits, dtype=float)
    n_workers = os.cpu_count() or 1
    if _decimal_uses_pool(x.size, scales.size, start_digits):
        chunks = np.array_split(np.arange(x.size), n_workers)
        args = [(x[c], scales, kind, start_digits[c], max_digits) for c in chunks]
        try:
            with concurrent.futures.ProcessPoolExecutor(n_workers) as pool:
                results = list(pool.map(_decimal_chunk, args))
        except (OSError, RuntimeError, concurrent.futures.process.BrokenProcessPool):
            results = None
        if results is not None:
            return np.concatenate([r[0] for r in results]), np.concatenate([r[1] for r in results])
    return _decimal_chunk((x, scales, kind, start_digits, max_digits))


# cost model (seconds, fitted on one machine; only ratios matter) used to split the points between
# the two uniformization variants and, in _choose_slow_method, against the Decimal method
_SERIES_FIXED_PER_K = 6.5e-6   # one step of the c_k recursion
_SERIES_PER_TERM = 4e-9        # one Poisson term for one point
_SERIES_PER_BLOCK = 5e-5       # fixed overhead of one block of points
_SQUARING_PER_POINT = 2e-6     # one point costs this ...
_SQUARING_PER_POINT_N3 = 1e-8  # ... plus this * (n+1)^3
_SERIES_MAX_LAMBDA = 1e5       # c_k rounding error grows like ~k * eps, keep it below ~1e-11
_SERIES_MAX_BLOCK_ENTRIES = 2 ** 22


def _series_blocks(lam_sorted, n):
    """Cut sorted lambdas into blocks [start, stop) whose range is <= 10 sqrt(lambda) so that one
    Poisson window of half-width 12 sqrt(lambda) + n + 30 around the block's mode covers all of
    them. Returns a list of (start, stop, half_width)."""
    blocks = []
    start = 0
    m = lam_sorted.size
    while start < m:
        lo = lam_sorted[start]
        stop = int(np.searchsorted(lam_sorted, lo + 10 * np.sqrt(lo + 1), side='right'))
        hi = lam_sorted[stop - 1]
        half = int(12 * np.sqrt(hi + 1) + n + 30)
        stop = min(stop, start + max(1, _SERIES_MAX_BLOCK_ENTRIES // (2 * half + 1)))
        blocks.append((start, stop, half))
        start = stop
    return blocks


def _phase_type_series(x, scales, kind):
    """Uniformization series, see _phase_type_method:

        value(x) = sum_k Poisson(k; lambda) c_k,   lambda = x mu,   c_k = (alpha P^k)[col],
        P = I + Q / mu.

    The c_k do not depend on x and are obtained by K vector-bidiagonal products, O(K n) in total
    with K ~ lambda_max. The points are sorted by lambda and cut into blocks (_series_blocks); in
    a block the Poisson weights are generated from the weight at the block's mode k0 by the
    recursions w_{k+1} = w_k lambda / (k + 1) and w_{k-1} = w_k k / lambda, as cumulative
    products over a window of indices, vectorized over both points and indices (no Python loop
    over k). Each sweep is extended, for the points that need it, until a rigorous bound on the
    remaining tail (0 <= c_k <= 1, geometric decay of the weights) is below eps times the value
    accumulated so far; typically ~24 sqrt(lambda) + 2n terms suffice. Finally the result is
    divided by the sum of the weights used (= 1 up to the negligible tails), which removes the
    rounding error of the starting weight exp(k0 log lambda - lambda - lgamma(k0 + 1)) whose
    argument is large for large lambda. Everything is nonnegative, so no cancellation.
    """
    # --- Step 0: the Markov chain -------------------------------------------------------------
    # States 0..n-1 are "currently in exponential phase i", state n is absorbing (sum finished).
    # Uniformization with rate mu = max(rate) turns the continuous-time chain into a discrete one
    # watched at the ticks of a Poisson(mu) clock: at each tick the chain in phase i moves to
    # phase i + 1 with probability move[i] = rate_i / mu and stays with probability stay[i].
    # Then  expm(x Q) = sum_k Poisson(k; x mu) P^k  (the number of ticks in [0, x] is
    # Poisson(x mu) and after k ticks the chain is distributed as alpha P^k).
    rates = 1.0 / scales
    n = rates.size
    mu = rates.max()
    stay = 1.0 - rates / mu
    move = rates / mu
    # pdf needs the probability of being in the last phase (then times rate_{n-1}, applied by the
    # caller), cdf the probability of having been absorbed
    col = n - 1 if kind == 'pdf' else n
    lam = x * mu
    out = np.empty(x.size)

    order = np.argsort(lam)
    lam_sorted = lam[order]
    lam_max = lam_sorted[-1]
    # Poisson weights beyond the mode + 40 sigma underflow to exactly 0, so the upward sweep
    # always terminates before K
    K = int(np.ceil(lam_max + 40 * np.sqrt(lam_max + 1) + n + 50))

    # --- Step 1: c_k = probability of being in state `col` after k ticks, k = 0..K -------------
    # v is the distribution over the n + 1 states after k ticks (v_0 = alpha = (1, 0, ..., 0)); one
    # tick is v <- v P with P bidiagonal, i.e. O(n) per step. All quantities are probabilities
    # (nonnegative, <= 1), so there is no cancellation.
    c = np.empty(K + 1)
    v = np.zeros(n + 1)
    v[0] = 1.0
    for k in range(K + 1):
        c[k] = v[col]
        nxt = np.empty(n + 1)
        nxt[:n] = v[:n] * stay                 # stayed in phase i
        nxt[1:n] += v[:n - 1] * move[:n - 1]   # moved from phase i - 1 to phase i
        nxt[n] = v[n] + v[n - 1] * move[n - 1]  # absorbed (stays absorbed)
        v = nxt

    # --- Step 2: for each block of points, sum_k Poisson(k; lambda) c_k -------------------------
    # A block holds points whose lambdas lie within ~10 sqrt(lambda) of each other, so that a
    # single set of Poisson indices around the block's smallest lambda (k0) serves them all. The
    # Poisson weights are generated from w0 = Poisson(k0; lambda) (computed in log space, the only
    # place where a transcendental function is evaluated per point) by the exact recurrences
    #     w_{k+1} = w_k * lambda / (k + 1)     and     w_{k-1} = w_k * k / lambda,
    # as cumulative products over windows of `half` indices, vectorized over the points in the
    # block. Sweeps go upward from k0 and downward from k0.

    for start, stop, half in _series_blocks(lam_sorted, n):
        idx = order[start:stop]
        lb = lam_sorted[start:stop]
        k0 = int(lb[0])
        w0 = np.exp(sp.special.xlogy(k0, lb) - lb - sp.special.gammaln(k0 + 1))
        acc = w0 * c[k0]
        wsum = w0.copy()

        # upward sweep: weights at k0+1 .. k_to. `active` are the points whose tail bound is not
        # yet negligible; a window is appended for them only.
        active = np.arange(lb.size)
        w_last = w0.copy()
        k_from, k_to = k0 + 1, min(K, k0 + half)
        while k_from <= k_to and active.size:
            ks = np.arange(k_from, k_to + 1)
            W = w_last[active, None] * np.cumprod(lb[active, None] / ks[None, :], axis=1)
            acc[active] += W @ c[ks]
            wsum[active] += W.sum(axis=1)
            w_last[active] = W[:, -1]
            # k_to > hi >= lb here, so the remaining weights decay at least geometrically with
            # ratio lambda / (k_to + 1) < 1; with c_k <= 1 the remaining sum is bounded by
            # w_last * (k_to + 1) / (k_to + 1 - lambda). Stop when that is below eps * acc.
            tail = w_last[active] * (k_to + 1) / (k_to + 1 - lb[active])
            active = active[tail > _EPS * acc[active]]
            k_from, k_to = k_to + 1, min(K, k_to + half)

        # downward sweep: weights at k0-1 .. k_to
        active = np.arange(lb.size)
        w_last = w0.copy()
        k_from, k_to = k0 - 1, max(0, k0 - half)
        while k_from >= k_to and active.size:
            ks = np.arange(k_from, k_to - 1, -1)
            W = w_last[active, None] * np.cumprod((ks[None, :] + 1) / lb[active, None], axis=1)
            acc[active] += W @ c[ks]
            wsum[active] += W.sum(axis=1)
            w_last[active] = W[:, -1]
            if k_to == 0:
                break
            # k_to < lo <= lb here: going down the weights decay with ratio k / lambda < 1, so the
            # remaining sum is bounded by w_last * lambda / (lambda - k_to)
            tail = w_last[active] * lb[active] / (lb[active] - k_to)
            active = active[tail > _EPS * acc[active]]
            k_from, k_to = k_to - 1, max(0, k_to - half)

        # --- Step 3: normalize ----------------------------------------------------------------
        # The weights used sum to 1 up to the (negligible) tails, so dividing by their sum cancels
        # the rounding error of w0 (its log-space argument is O(lambda), hence an absolute error
        # ~lambda * eps that would otherwise be inherited by every weight).
        out[idx] = acc / wsum
    return out


def _phase_type_squaring(x, scales, kind, max_block_entries=2 ** 22):
    """Scaling-and-squaring version of the uniformization, chosen by the cost model in _phase_type_method.
    O(len(x) * (len(scales) + log2(x mu)) * len(scales)^3), see _phase_type_method."""
    rates = 1.0 / scales
    n = rates.size
    mu = rates.max()
    # B = Q + mu I is nonnegative (bidiagonal: mu - rate_i on the diagonal, rate_i above it, and
    # mu for the absorbing state) with row sums mu, so expm(h Q) = exp(-h mu) expm(h B) and the
    # Taylor series of expm(h B) has only nonnegative terms.
    idx = np.arange(n)
    B = np.zeros((n + 1, n + 1))
    B[idx, idx] = mu - rates
    B[idx, idx + 1] = rates
    B[n, n] = mu

    col = n - 1 if kind == 'pdf' else n
    out = np.empty(x.size)

    # Step 1: choose s so that h = x / 2^s has h mu <= 1; then the Taylor series of expm(h B)
    # converges in ~n + 20 terms (the k-th term is <= 1/k! in norm) and expm(x Q) = expm(h Q)^(2^s).
    squarings = np.zeros(x.size, dtype=int)
    big = x * mu > 1
    squarings[big] = np.ceil(np.log2(x[big] * mu)).astype(int)

    # Points sharing the same s are processed together as a stack of (n+1)x(n+1) matrices (batched
    # matmul), in blocks small enough to keep memory bounded.
    block = max(1, max_block_entries // (n + 1) ** 2)
    for s in np.unique(squarings):
        where = np.flatnonzero(squarings == s)
        for start in range(0, where.size, block):
            sel = where[start:start + block]
            h = x[sel] / 2.0 ** s
            A = h[:, None, None] * B
            # Step 2: Taylor series E = sum_k A^k / k!, all terms nonnegative; stop once the newest
            # term no longer changes any entry (relative eps), but not before k = n because the
            # first n powers of a bidiagonal matrix are the ones that fill in the corner entries
            E = np.broadcast_to(np.eye(n + 1), A.shape).copy()
            term = E.copy()
            for k in range(1, n + 100):
                term = term @ A / k
                E += term
                if k >= n and np.all(term <= _EPS * E):
                    break
            # Step 3: expm(h Q) = exp(-h mu) E, then square s times to get expm(x Q)
            E *= np.exp(-h * mu)[:, None, None]
            for _ in range(s):
                E = E @ E
                # E is stochastic; renormalizing the rows stops the row-sum rounding error from
                # doubling at every squaring (2^s eps otherwise)
                E /= E.sum(axis=2, keepdims=True)
            out[sel] = E[:, 0, col]
    return out


def _phase_type_plan(lam, n):
    """Decide which lambdas = x mu the series variant should handle (those <= threshold; the
    others go to scaling and squaring) by minimizing the modelled cost over a few candidate
    thresholds. Returns (threshold, predicted_total_seconds)."""
    lam_sorted = np.sort(lam)
    m = lam_sorted.size
    sq_point = _SQUARING_PER_POINT + _SQUARING_PER_POINT_N3 * (n + 1) ** 3
    per_point = np.cumsum(_SERIES_PER_TERM * (34 * np.sqrt(lam_sorted) + 2 * n + 60))
    best_cost, threshold = m * sq_point, -1.0
    candidates = np.unique(np.minimum(m, np.geomspace(1, m, 24).astype(int)))
    for j in candidates:
        L = lam_sorted[j - 1]
        if L > _SERIES_MAX_LAMBDA:
            break
        n_blocks = len(_series_blocks(lam_sorted[:j], n))
        cost = (_SERIES_FIXED_PER_K * (L + 40 * np.sqrt(L + 1) + n + 50) + per_point[j - 1]
                + _SERIES_PER_BLOCK * n_blocks + (m - j) * sq_point)
        if cost < best_cost:
            best_cost, threshold = cost, L
    return threshold, best_cost


def _phase_type_method(x, scales, kind):
    """Phase-type representation, valid for any scales including repeated ones.
    https://en.wikipedia.org/wiki/Hypoexponential_distribution#Relation_to_the_phase-type_distribution

    The distribution is the absorption time of the Markov chain 0 -> 1 -> ... -> n (state i leaves
    at rate 1/scale_i, state n absorbing) with generator Q, so with alpha = (1, 0, ..., 0)
        pdf(x) = rate_{n-1} * expm(x Q)[0, n-1],   cdf(x) = expm(x Q)[0, n].
    expm(x Q) is evaluated by uniformization rather than scipy.linalg.expm: with mu = max(rate)
    and P = I + Q / mu (a nonnegative, bidiagonal stochastic matrix),
        expm(x Q) = exp(-x mu) * sum_k (x mu)^k / k! * P^k,
    a sum of nonnegative terms, so no cancellation ever occurs and every entry is accurate to
    ~machine precision, including tiny values for very small or very large x (scipy's expm loses
    relative accuracy at small x, e.g. 2e-4 relative error at x = 1e-6 for scales [1, 1, 2]).

    Two variants, with lambda = x mu:
      * _phase_type_series sums the series directly: a fixed O(lambda_max n) part plus
        O(sqrt(lambda) + n) per point, i.e. essentially linear in both len(x) and len(scales).
        Rounding error ~ lambda eps, so it is only used for lambda <= 1e5.
      * _phase_type_squaring uses scaling and squaring of expm(h Q), h mu <= 1:
        O((n + log2 lambda) n^3) per point, accurate to ~n eps for any lambda.
    The points are split between the two by a small cost model (small n and large lambda favour
    squaring, large n or many points favour the series).
    """
    x = np.asarray(x, dtype=float)
    n = scales.size
    mu = (1.0 / scales).max()
    lam = x * mu
    out = np.empty(x.size)

    threshold, _ = _phase_type_plan(lam, n)
    use_series = lam <= threshold
    if np.any(use_series):
        out[use_series] = _phase_type_series(x[use_series], scales, kind)
    if not np.all(use_series):
        out[~use_series] = _phase_type_squaring(x[~use_series], scales, kind)
    if kind == 'pdf':
        out *= 1.0 / scales[-1]
    else:
        np.minimum(out, 1.0, out=out)
    return out


def _is_uniform_grid_from_zero(x):
    """True if x is a 1-D grid 0, dx, 2dx, ... (the only domain on which the discrete convolution
    of the exponential pdfs is the hypoexponential pdf)."""
    if x.ndim != 1 or x.size < 2 or x[0] != 0:
        return False
    steps = np.diff(x)
    return steps[0] > 0 and np.allclose(steps, steps[0], rtol=1e-8, atol=0)


def _convolution_method(x, scales):
    """Numerical convolution of the len(scales) exponential pdfs on a uniform grid starting at 0.

    Fast (FFT based) even for very long grids, but it is a discretization of the convolution
    integral so its accuracy depends on the grid spacing dx. The discrete convolution approximates
    the integral with the trapezoidal rule (sum times dx, half weight on the two end samples), which
    makes the error O(dx^2). Note: the old normalization divided by sum(window) instead of
    multiplying by dx; the two agree only when the grid is long enough to contain almost all the
    mass of the window, which is why the pdf came out 58% too high on [0, 1] for scales [1, 1].
    """
    dx = x[1] - x[0]
    pdf = sp.stats.expon.pdf(x, scale=scales[0])
    for s in scales[1:]:
        window = sp.stats.expon.pdf(x, scale=s)
        full = sp.signal.convolve(pdf, window, mode='full')[:x.size]
        full -= 0.5 * (pdf[0] * window + pdf * window[0])
        pdf = dx * full
    return np.maximum(pdf, 0.0)


# Empirical cost model (seconds) for the two slow, exact fallbacks, fitted on one machine (only
# the ratio of the two predictions matters; see the benchmark in the notebook). m = number of
# points, n = len(scales), d = Decimal precision in digits. Decimal is a pure-Python loop: ~linear in
# m and n (plus the O(n^2) weights per precision level) and growing with d; it can only be sped up
# by spreading points over processes. The phase-type method is vectorized numpy with a fixed
# overhead and a per-point cost that is ~linear in n (see _phase_type_plan for its own model). In
# practice Decimal only wins for a handful of points with few scales, but the model keeps the
# choice data driven.
_DECIMAL_PER_TERM = 1.8e-5     # one exp + products at 28 digits; grows like sqrt(digits)
_DECIMAL_PER_WEIGHT = 4e-6     # one of the n^2 ratio products, per precision level; linear in digits
_DECIMAL_POOL_OVERHEAD = 0.05  # starting the worker processes
_DECIMAL_POOL_EFFICIENCY = 0.8
_PHASE_TYPE_OVERHEAD = 2e-4    # fixed cost of one phase-type call


def _decimal_serial_cost(m, n, digits):
    d = np.broadcast_to(np.asarray(digits, dtype=float), (m,)) / 28.0
    levels = np.unique(d).size
    return _DECIMAL_PER_TERM * n * np.sum(np.sqrt(d)) + levels * n * n * _DECIMAL_PER_WEIGHT * np.mean(d)


def _decimal_uses_pool(m, n, digits):
    workers = os.cpu_count() or 1
    return workers > 1 and m >= 2 * workers and _decimal_serial_cost(m, n, digits) > _DECIMAL_PARALLEL_MIN_COST


def _predicted_cost(method, m, n, digits, lam=None):
    """Predicted seconds for one of the exact fallbacks on m points with n scales. `digits` is the
    Decimal starting precision (scalar or one value per point); `lam` = x * max(rate) is needed
    for the phase-type method, whose cost depends on how far in the tail the points lie."""
    if method == 'decimal':
        serial = _decimal_serial_cost(m, n, digits)
        if _decimal_uses_pool(m, n, digits):
            return _DECIMAL_POOL_OVERHEAD + serial / (_DECIMAL_POOL_EFFICIENCY * (os.cpu_count() or 1))
        return serial
    lam = np.ones(m) if lam is None else np.asarray(lam, dtype=float)
    return _PHASE_TYPE_OVERHEAD + _phase_type_plan(lam, n)[1]


def _choose_slow_method(x, scales, digits, tolerance):
    """Pick the exact fallback for the points x. Decimal when the caller asks for more accuracy
    than double precision can deliver (the phase-type method is accurate to ~1e-13 relative),
    otherwise whichever the cost model predicts to be faster. Empirically (see the notebook)
    Decimal only wins for a handful of points with few scales and moderate precision; the
    vectorized phase-type method wins for many points, and for many scales because its cost is
    ~linear in len(scales) while Decimal's weights alone cost O(len(scales)^2)."""
    if tolerance < 1e-12:
        return 'decimal'
    m, n = x.size, scales.size
    lam = x * (1.0 / scales).max()
    if _predicted_cost('decimal', m, n, digits) < _predicted_cost('phase_type', m, n, digits, lam):
        return 'decimal'
    return 'phase_type'


def _start_digits(rel_err):
    """Decimal starting precision for points whose float closed form had relative error bound
    `rel_err`: 17 significant digits are wanted, and the cancellation destroys log10(rel_err / eps)
    of them, so start there (28 = Decimal default at least; capped at the Decimal max_digits)."""
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        digits = 17 + np.log10(rel_err / _EPS)
    digits = np.where(np.isfinite(digits), digits, 60)
    return np.clip(np.ceil(digits), 28, 2000)


def _slow(x, scales, kind, method, start_digits):
    """Run one of the exact fallbacks ('decimal' or 'phase_type') on all points of x."""
    if method == 'decimal':
        value, converged = _decimal_method(x, scales, kind, start_digits)
        if not np.all(converged):
            value[~converged] = _phase_type_method(x[~converged], scales, kind)
        return value
    return _phase_type_method(x, scales, kind)


def _pointwise(x, scales, kind, tolerance, method):
    """Cascade for arbitrary points x > 0 (finite): float closed form, then for the points that
    fail its error bound either the Decimal closed form or the phase-type method, chosen by the
    cost model. Repeated scales go straight to the phase-type method."""
    distinct = np.unique(scales).size == scales.size
    if method in ('closed_form', 'decimal') and not distinct:
        raise ValueError(f"method={method!r} requires distinct scale parameters.")
    n = scales.size

    if method == 'phase_type' or not distinct:
        return _phase_type_method(x, scales, kind)

    value, rel_err = _closed_form_method(x, scales, kind)
    if method == 'closed_form':
        return value
    if method == 'decimal':
        return _slow(x, scales, kind, 'decimal', _start_digits(rel_err))

    good = np.isfinite(value) & (value > 0) & (rel_err <= tolerance)
    res = np.where(good, value, 0.0)
    todo = np.flatnonzero(~good)
    if todo.size:
        digits = _start_digits(rel_err[todo])
        slow_method = _choose_slow_method(x[todo], scales, digits, tolerance)
        res[todo] = _slow(x[todo], scales, kind, slow_method, digits)
    return res


def _sanity_failures(x, values, kind, scales, tolerance):
    """Cheap common-sense checks on a finished result (x: finite, > 0, 1-D; values aligned).
    Returns the list of failed checks (empty if everything looks right)."""
    failures = []
    if not np.all(np.isfinite(values)):
        failures.append('non-finite values')
    if np.any(values < 0):
        failures.append('negative values')
    if kind == 'cdf' and np.any(values > 1 + 1e-12):
        failures.append('cdf above 1')
    if kind == 'pdf' and scales.size > 1 and np.any(values > 1.0 / scales.min()):
        failures.append('pdf above the bound 1/min(scales)')

    if x.size >= 3:
        order = np.argsort(x, kind='stable')
        xs, vs = x[order], values[order]
        significant = np.abs(np.diff(vs)) > 1e-12 * np.max(np.abs(vs))
        signs = np.sign(np.diff(vs))[significant]
        if kind == 'cdf':
            if np.any(signs < 0):
                failures.append('cdf not monotone')
        else:
            # The hypoexponential pdf is log-concave, hence unimodal: at most one rise->fall switch.
            switches = np.count_nonzero(np.diff(signs) != 0)
            if switches > 1:
                failures.append('pdf not unimodal')

        # If the grid is fine and covers the distribution, the pdf must integrate to 1.
        if kind == 'pdf' and x.size >= 1000:
            mean, std = scales.sum(), np.sqrt(np.sum(scales ** 2))
            steps = np.diff(xs)
            covers = xs[0] <= 1e-3 * mean and xs[-1] >= mean + 8 * std
            fine = steps.max() <= 0.05 * std
            if covers and fine:
                mass = sp.integrate.simpson(vs, x=xs)
                if abs(mass - 1) > max(1e-3, tolerance):
                    failures.append(f'pdf integrates to {mass:.6f} instead of 1')
    return failures


def _evaluate(x, scales, kind, tolerance, method):
    """Shared driver of hypoexp_pdf / hypoexp_cdf:
      1. validate arguments; handle x < 0, x = 0, x = inf and nan directly;
      2. pdf on a long uniform grid from 0 -> try the convolution, validated against exact values;
      3. otherwise (or if that fails) the pointwise cascade of _pointwise;
      4. common-sense checks on the result (_sanity_failures); with method='auto' a failure
         recomputes everything with the exact methods, with a forced method it only warns;
      5. reshape to the shape of x (a Python float for scalar x)."""
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}.")
    if kind == 'cdf' and method == 'convolution':
        raise ValueError("method='convolution' is only available for the pdf.")
    scales = _as_scales(scales)
    n = scales.size
    x_arr = np.asarray(x, dtype=float)
    xf = x_arr.ravel()

    out = np.full(xf.shape, np.nan)
    out[xf < 0] = 0.0
    out[xf == np.inf] = 0.0 if kind == 'pdf' else 1.0
    out[xf == 0] = (1.0 / scales[0] if n == 1 else 0.0) if kind == 'pdf' else 0.0
    valid = np.isfinite(xf) & (xf > 0)
    xv = xf[valid]

    used = method
    computed = None
    use_convolution = kind == 'pdf' and n > 1 and (
        method == 'convolution'
        or (method == 'auto' and x_arr.size >= _CONVOLUTION_MIN_POINTS and _is_uniform_grid_from_zero(x_arr)))
    if use_convolution:
        if not _is_uniform_grid_from_zero(x_arr):
            raise ValueError("method='convolution' requires a 1-D uniformly spaced grid starting at 0.")
        conv = _convolution_method(x_arr, scales)
        if method == 'convolution':
            computed = conv[valid]
        else:
            # Validate the discretization against exact values at a few points (grid independent,
            # unlike checking that the pdf integrates to 1) and fall back to the pointwise methods
            # if the convolution is not accurate enough.
            check = np.unique(np.concatenate([[np.argmax(conv)], np.linspace(1, x_arr.size - 1, 15).astype(int)]))
            exact = _pointwise(x_arr[check], scales, 'pdf', tolerance, 'auto')
            if np.all(np.abs(conv[check] - exact) <= tolerance * conv.max()):
                computed, used = conv[valid], 'convolution'
            else:
                warnings.warn("Convolution on this grid was not accurate enough (grid too coarse); falling back to pointwise methods.")

    if computed is None and xv.size:
        computed = _pointwise(xv, scales, kind, tolerance, method)

    if xv.size:
        # Final common-sense checks. With method='auto' a failure triggers a recomputation of all
        # points with an exact method (phase-type first, then Decimal if scales are distinct);
        # with a forced method we only warn, since the caller asked for that method's raw output.
        failures = _sanity_failures(xv, computed, kind, scales, tolerance)
        if failures and method == 'auto':
            distinct = np.unique(scales).size == scales.size
            retries = ['phase_type'] + (['decimal'] if distinct else [])
            if used == 'phase_type':
                retries.remove('phase_type')
            for retry in retries:
                computed = _slow(xv, scales, kind, retry, np.full(xv.size, 60.0))
                used = retry
                failures = _sanity_failures(xv, computed, kind, scales, tolerance)
                if not failures:
                    break
        if failures:
            warnings.warn(f"hypoexp_{kind} sanity checks failed ({'; '.join(failures)}); returning the best available result.")
        out[valid] = computed

    return float(out[0]) if x_arr.ndim == 0 else out.reshape(x_arr.shape)


def hypoexp_pdf(x: ArrayLike, scales: ArrayLike, tolerance: float = 1e-6, method: str = 'auto') -> Union['np.ndarray', float]:
    """
    Compute the probability density function (PDF) of a hypoexponential distribution.

    The hypoexponential distribution is the distribution of the sum of independent
    exponential random variables with potentially different (possibly repeated) scales.

    Parameters
    ----------
    x : array_like
        Quantiles, where the PDF will be evaluated. Scalars, any shape and any spacing are fine.
    scales : array_like
        A single scale parameter or an iterable of scale parameters (must be > 0).
    tolerance : float, optional
        Accuracy requirement used to accept the fast methods (default 1e-6): the float closed form
        is accepted at a point if its relative rounding error bound is below `tolerance`; the
        convolution is accepted if it matches exact values to within `tolerance * max(pdf)`.
    method : {'auto', 'closed_form', 'decimal', 'convolution', 'phase_type'}, optional
        'auto' (default) uses the cascade described below; the others force a single method
        ('closed_form' and 'decimal' need distinct scales, 'convolution' needs a uniform grid
        starting at 0 and at least two scales).

    Returns
    -------
    pdf : ndarray or scalar
        Probability density function evaluated at x.

    Notes
    -----
    Method selection with method='auto', fastest first, falling back only where needed:

    1. If `x` is a uniform grid starting at 0 with >= 1e5 points, the FFT convolution is used and
       validated against exact values at a few grid points.
    2. Otherwise (or if the convolution fails validation), for distinct scales the float closed
       form is used at every point whose cancellation error bound is below `tolerance`.
    3. Points that fail are recomputed with an exact method: the Decimal closed form (precision
       chosen from the error bound and raised until a rigorous bound proves the result) or the
       phase-type method (uniformization, no cancellation), whichever a cost model predicts to be
       faster for that many points and scales.
    4. Repeated scales, and points where Decimal did not converge, use the phase-type method.
    5. Before returning, cheap sanity checks (finite, non-negative, pdf <= 1/min(scales),
       unimodal pdf / monotone cdf <= 1, pdf integrates to 1 on a fine covering grid) are run; a
       failure triggers a recomputation of all points with the exact methods.
    """
    return _evaluate(x, scales, 'pdf', tolerance, method)


def hypoexp_cdf(x: ArrayLike, scales: ArrayLike, tolerance: float = 1e-6, method: str = 'auto') -> Union['np.ndarray', float]:
    """
    Compute the cumulative distribution function (CDF) of a hypoexponential distribution.

    Parameters are as for `hypoexp_pdf` and the same closed form -> Decimal -> phase-type cascade
    is used (there is no convolution method for the CDF).
    """
    return _evaluate(x, scales, 'cdf', tolerance, method)


class Hypoexponential:
    """
    Object-oriented interface for the Hypoexponential distribution, parametrized by rates `eta`.
    Repeated and nearly equal rates are supported.
    """
    def __init__(self, eta):
        # Only validate and store; the closed-form weights used to be computed here, which raised
        # ZeroDivisionError for repeated rates. They are now computed lazily (see `weights`) and
        # pdf/cdf go through hypoexp_pdf/hypoexp_cdf, which handle repeated rates.
        rates = np.atleast_1d(np.asarray(eta, dtype=float)).ravel()
        if rates.size == 0 or not np.all(np.isfinite(rates)) or np.any(rates <= 0):
            raise ValueError("Rates must be finite and strictly greater than 0.")
        self._eta = rates
        self._scales = 1.0 / rates

    def pdf(self, x):
        return hypoexp_pdf(x, self._scales)

    def cdf(self, x):
        return hypoexp_cdf(x, self._scales)

    @property
    def weights(self):
        """Closed-form mixture weights prod_{j != i} 1 / (1 - eta_i / eta_j); only defined for distinct rates."""
        if np.unique(self._eta).size != self._eta.size:
            raise ValueError("Closed-form weights are undefined when rates are repeated.")
        weights = []
        for i, eta_i in enumerate(self._eta):
            others = np.delete(self._eta, i)
            weights.append(float(np.prod(1.0 / (1.0 - eta_i / others))))
        return weights

    @property
    def params(self):
        return {'rates': self._eta.tolist()}

    def sample(self, n_sample=1):
        return hypoexp_rvs(self._scales, size=n_sample)


if __name__ == '__main__':
    from numpy import linspace
    import matplotlib.pyplot as plt

    eta = [0.05665075, 0.07182203, 0.05739665, 0.05739665]

    sum_exp = Hypoexponential(eta)
    sample = sum_exp.sample(1000)
    t_range = linspace(0, max(sample), 500)
    plt.hist(sample, bins=50, density=True)
    plt.plot(t_range, sum_exp.pdf(t_range))
    plt.show()

    print(sum_exp.params)
    plt.plot(t_range, sum_exp.cdf(t_range))
    plt.show()
