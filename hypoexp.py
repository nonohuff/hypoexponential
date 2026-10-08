import numpy as np
import scipy as sp
from typing import Union
from decimal import Decimal, localcontext
import warnings

ArrayLike = Union[int, float, list, 'np.ndarray']

_EPS = np.finfo(float).eps
_METHODS = ('auto', 'closed_form', 'decimal', 'convolution', 'phase_type')
_CONVOLUTION_MIN_POINTS = 10 ** 5


def _as_scales(scales: ArrayLike) -> 'np.ndarray':
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
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        ratio = scales[:, None] / (scales[:, None] - scales[None, :])
        np.fill_diagonal(ratio, 1.0)
        weights = np.prod(ratio, axis=1)
        if kind == 'pdf':
            basis = np.exp(-x[None, :] / scales[:, None]) / scales[:, None]
        else:
            basis = -np.expm1(-x[None, :] / scales[:, None])
        terms = weights[:, None] * basis
        value = np.sum(terms, axis=0)
        rel_err = 4 * n * _EPS * np.sum(np.abs(terms), axis=0) / np.abs(value)
    return value, rel_err


def _decimal_method(x, scales, kind, start_digits, max_digits=2000):
    """Closed form evaluated with the `decimal` module (distinct scales only).

    Arbitrary precision removes the cancellation problem of the float closed form, at the cost of
    speed (pure Python loops, cost grows with len(scales) and with the precision needed). For each
    point the precision starts at `start_digits` (chosen from the float error bound) and doubles
    until two consecutive evaluations agree to 1e-13 relative, which proves convergence; this
    replaces the old "integrate to 1" criterion that could never be met for a single point or a
    truncated grid. Points that still have not converged at `max_digits` are reported as failed.

    Returns (value, converged_mask).
    """
    scales_dec = [Decimal(float(s)) for s in scales]
    out = np.empty(x.size)
    ok = np.zeros(x.size, dtype=bool)
    weights_cache = {}

    def weights(prec):
        if prec not in weights_cache:
            ws = []
            for i, s_i in enumerate(scales_dec):
                w = Decimal(1)
                for j, s_j in enumerate(scales_dec):
                    if i != j:
                        w *= s_i / (s_i - s_j)
                ws.append(w)
            weights_cache[prec] = ws
        return weights_cache[prec]

    with localcontext() as ctx:
        for k, x_k in enumerate(x):
            x_dec = Decimal(float(x_k))
            prec = int(start_digits[k])
            previous = None
            while prec <= max_digits:
                ctx.prec = prec
                total = Decimal(0)
                for w, s_i in zip(weights(prec), scales_dec):
                    e = (-x_dec / s_i).exp()
                    total += w * e / s_i if kind == 'pdf' else w * (1 - e)
                value = float(total)
                if previous is not None and value > 0 and abs(value - previous) <= 1e-13 * value:
                    out[k], ok[k] = value, True
                    break
                previous = value
                prec *= 2
            else:
                out[k] = previous
    return out, ok


def _phase_type_method(x, scales, kind, max_block_entries=2 ** 22):
    """Phase-type representation, valid for any scales including repeated ones.
    https://en.wikipedia.org/wiki/Hypoexponential_distribution#Relation_to_the_phase-type_distribution

    The distribution is the absorption time of the Markov chain 0 -> 1 -> ... -> n (state i leaves
    at rate 1/scale_i, state n absorbing) with generator Q, so with alpha = (1, 0, ..., 0):
        pdf(x) = rate_{n-1} * expm(x Q)[0, n-1],   cdf(x) = expm(x Q)[0, n].
    The matrix exponential is evaluated by uniformization rather than scipy.linalg.expm: with
    mu = max(rate) and h = x / 2**s chosen so that h*mu <= 1,
        expm(h Q) = exp(-h mu) * expm(h (Q + mu I)),
    where h (Q + mu I) is a nonnegative matrix with row sums <= 1. Its Taylor series has only
    nonnegative terms and the s squarings that give expm(x Q) only multiply and add nonnegative
    matrices, so no cancellation ever occurs and every entry is accurate to ~machine precision,
    including tiny values for very small or very large x (scipy's expm is accurate for moderate x
    but loses relative accuracy at small x, e.g. 2e-4 relative error at x = 1e-6 for scales
    [1, 1, 2]). Cost is O(len(x) * (len(scales) + log2(x*mu)) * len(scales)^3), so this is the
    slowest float method for many points and large len(scales), but it is fully vectorized over x.
    """
    rates = 1.0 / scales
    n = rates.size
    mu = rates.max()
    idx = np.arange(n)
    B = np.zeros((n + 1, n + 1))
    B[idx, idx] = mu - rates
    B[idx, idx + 1] = rates
    B[n, n] = mu

    col = n - 1 if kind == 'pdf' else n
    out = np.empty(x.size)

    squarings = np.zeros(x.size, dtype=int)
    big = x * mu > 1
    squarings[big] = np.ceil(np.log2(x[big] * mu)).astype(int)

    block = max(1, max_block_entries // (n + 1) ** 2)
    for s in np.unique(squarings):
        where = np.flatnonzero(squarings == s)
        for start in range(0, where.size, block):
            sel = where[start:start + block]
            h = x[sel] / 2.0 ** s
            A = h[:, None, None] * B
            E = np.broadcast_to(np.eye(n + 1), A.shape).copy()
            term = E.copy()
            for k in range(1, n + 100):
                term = term @ A / k
                E += term
                if k >= n and np.all(term <= _EPS * E):
                    break
            E *= np.exp(-h * mu)[:, None, None]
            for _ in range(s):
                E = E @ E
            out[sel] = E[:, 0, col]

    if kind == 'pdf':
        out *= rates[-1]
    return out


def _is_uniform_grid_from_zero(x):
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


# Empirical cost model (seconds) for the two slow, exact fallbacks, fitted on this machine
# (see the "Decimal vs phase-type" section of the notebook). m = number of points, n = len(scales),
# d = starting Decimal precision in digits. Decimal is a pure-Python loop: linear in m and n, and
# superlinear in d. The phase-type method is vectorized: a fixed overhead plus a per-point cost that
# is tiny for small n and grows like (n + 1)^3. In practice Decimal only wins for 1-2 points with
# few scales and modest precision, but the model keeps the choice data driven.
_DECIMAL_COST_PER_POINT = 3e-5
_PHASE_TYPE_OVERHEAD = 1.5e-4
_PHASE_TYPE_COST_PER_POINT = 1e-6
_PHASE_TYPE_COST_PER_POINT_N3 = 4e-9


def _predicted_cost(method, m, n, digits):
    if method == 'decimal':
        return m * n * _DECIMAL_COST_PER_POINT * (digits / 28.0) ** 1.5
    return _PHASE_TYPE_OVERHEAD + m * (_PHASE_TYPE_COST_PER_POINT + _PHASE_TYPE_COST_PER_POINT_N3 * (n + 1) ** 3)


def _choose_slow_method(m, n, digits, tolerance):
    """Pick the exact fallback for m points. Decimal when the caller asks for more accuracy than
    double precision can deliver (the phase-type method is accurate to ~1e-14 relative), otherwise
    whichever the cost model predicts to be faster."""
    if tolerance < 1e-12:
        return 'decimal'
    if _predicted_cost('decimal', m, n, digits) < _predicted_cost('phase_type', m, n, digits):
        return 'decimal'
    return 'phase_type'


def _start_digits(rel_err):
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        digits = 17 + np.log10(rel_err / _EPS)
    digits = np.where(np.isfinite(digits), digits, 60)
    return np.clip(np.ceil(digits), 28, 400)


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
        slow_method = _choose_slow_method(todo.size, n, float(np.median(digits)), tolerance)
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
    3. Points that fail are recomputed with the Decimal closed form, increasing the precision
       until the result has converged.
    4. Repeated scales, and points where Decimal did not converge, use the phase-type method.
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
