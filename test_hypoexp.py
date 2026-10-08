import time
import warnings

import mpmath
import numpy as np
import pytest

from hypoexp import Hypoexponential, hypoexp_cdf, hypoexp_pdf, hypoexp_rvs

mpmath.mp.dps = 500


def _reference(x, scales, kind):
    """High-precision closed form; exactly repeated scales are split by 1e-60 (effect far below double precision)."""
    s = [mpmath.mpf(float(v)) + i * mpmath.mpf('1e-60') for i, v in enumerate(scales)]
    x = mpmath.mpf(float(x))
    total = mpmath.mpf(0)
    for i, si in enumerate(s):
        w = mpmath.mpf(1)
        for j, sj in enumerate(s):
            if i != j:
                w *= si / (si - sj)
        total += w * (mpmath.exp(-x / si) / si if kind == 'pdf' else -mpmath.expm1(-x / si))
    return float(total)


SCALES = [
    [2.0],
    [1.0, 2.0],
    [1.0, 1.0],
    [1.0, 1.0, 2.0],
    [1.0, 1.00000001, 1.00000002],
    [0.5, 1.0, 2.0, 4.0, 8.0],
    list(1.0 + 1e-6 * np.arange(10)),
    [3.0] * 6,
    [1e-3, 1.0, 1e3],
]
X = [0.0, 1e-6, 0.1, 1.0, 4.0, 25.0, 200.0]


@pytest.mark.parametrize("scales", SCALES)
@pytest.mark.parametrize("kind,func", [('pdf', hypoexp_pdf), ('cdf', hypoexp_cdf)])
def test_auto_matches_high_precision_reference(scales, kind, func):
    got = func(np.array(X), scales)
    for xi, gi in zip(X, got):
        ref = _reference(xi, scales, kind)
        assert gi >= 0
        if ref < 1e-290:
            assert gi < 1e-280
        else:
            assert gi == pytest.approx(ref, rel=1e-9), (xi, scales)
        # scalar call agrees with the array call (the fallback chosen may differ, so allow rounding-level differences)
        assert func(xi, scales) == pytest.approx(gi, rel=1e-13, abs=1e-300)


@pytest.mark.parametrize("method", ['decimal', 'phase_type'])
@pytest.mark.parametrize("scales", [[1.0, 2.0], [1.0, 1.00000001, 1.00000002], list(1.0 + 1e-6 * np.arange(6))])
def test_forced_slow_methods_match_reference(method, scales):
    x = np.array([1e-6, 0.5, 3.0, 40.0])
    for kind, func in [('pdf', hypoexp_pdf), ('cdf', hypoexp_cdf)]:
        got = func(x, scales, method=method)
        for xi, gi in zip(x, got):
            assert gi == pytest.approx(_reference(xi, scales, kind), rel=1e-10)


def test_closed_form_is_used_when_well_conditioned_and_fails_when_not():
    x = np.linspace(0.1, 20, 50)
    np.testing.assert_array_equal(hypoexp_pdf(x, [1, 2, 3]), hypoexp_pdf(x, [1, 2, 3], method='closed_form'))
    assert hypoexp_pdf(1.0, [1.0, 1.00000001, 1.00000002], method='closed_form') < 0  # garbage, as in Kevin's report
    assert hypoexp_pdf(1.0, [1.0, 1.00000001, 1.00000002]) == pytest.approx(0.1839397169069268, rel=1e-12)


def test_repeated_rates_in_class():
    dist = Hypoexponential([1, 1])
    assert dist.pdf(1.0) == pytest.approx(np.exp(-1), rel=1e-14)
    assert dist.cdf(1.0) == pytest.approx(1 - 2 * np.exp(-1), rel=1e-14)
    assert dist.params == {'rates': [1.0, 1.0]}
    with pytest.raises(ValueError):
        dist.weights


def test_convolution_normalization_on_short_grid():
    x = np.linspace(0, 1, 100000)
    conv = hypoexp_pdf(x, [1, 1], method='convolution')
    np.testing.assert_allclose(conv, x * np.exp(-x), rtol=0, atol=1e-9)
    assert conv[-1] == pytest.approx(np.exp(-1), rel=1e-9)


def test_auto_uses_convolution_on_fine_grid_and_falls_back_on_coarse_grid():
    fine = np.linspace(0, 30, 100000)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pdf = hypoexp_pdf(fine, [1, 2, 3])
    np.testing.assert_allclose(pdf, hypoexp_pdf(fine, [1, 2, 3], method='closed_form'), rtol=0, atol=1e-8)

    coarse = np.linspace(0, 1e5, 100000)  # dx = 1: convolution is far off, must be rejected
    with pytest.warns(UserWarning, match="Convolution"):
        pdf = hypoexp_pdf(coarse, [1, 2])
    np.testing.assert_allclose(pdf, np.exp(-coarse / 2) - np.exp(-coarse), rtol=1e-12, atol=1e-300)


def test_narrow_grid_returns_quickly_and_correctly():
    x = np.linspace(0, 4, 50)
    start = time.perf_counter()
    pdf = hypoexp_pdf(x, [1, 2])
    assert time.perf_counter() - start < 1
    np.testing.assert_allclose(pdf, np.exp(-x / 2) - np.exp(-x), rtol=1e-12, atol=1e-300)
    assert hypoexp_cdf(4.0, [1, 2]) == pytest.approx(0.747645, abs=1e-6)


def test_result_does_not_depend_on_grid():
    expected = hypoexp_pdf(2.0, [1.0, 1.00000001, 1.00000002])
    for grid in (np.array([2.0]), np.linspace(0, 2, 3), np.linspace(1.5, 2, 100000)[::-1], np.array([[7.0, 2.0]])):
        at_two = list(grid.flat).index(2.0)
        assert hypoexp_pdf(grid, [1.0, 1.00000001, 1.00000002]).flat[at_two] == pytest.approx(expected, rel=1e-12)


def test_pdf_integrates_to_cdf():
    x = np.linspace(0, 40, 20001)
    for scales in ([1.0, 1.0, 2.0], [0.3, 0.30001, 5.0]):
        from scipy.integrate import cumulative_trapezoid
        integral = cumulative_trapezoid(hypoexp_pdf(x, scales), x, initial=0)
        np.testing.assert_allclose(integral, hypoexp_cdf(x, scales), atol=1e-7, rtol=0)


def test_edge_inputs_and_shape():
    x = np.array([[-1.0, 0.0], [np.inf, np.nan]])
    pdf, cdf = hypoexp_pdf(x, [1, 2]), hypoexp_cdf(x, [1, 2])
    assert pdf.shape == cdf.shape == (2, 2)
    assert pdf[0, 0] == 0 and pdf[0, 1] == 0 and pdf[1, 0] == 0 and np.isnan(pdf[1, 1])
    assert cdf[0, 0] == 0 and cdf[0, 1] == 0 and cdf[1, 0] == 1 and np.isnan(cdf[1, 1])
    assert hypoexp_pdf(0.0, [2.0]) == 0.5
    assert isinstance(hypoexp_pdf(1.0, [1, 2]), float)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        hypoexp_pdf(1.0, [1.0, 0.0])
    with pytest.raises(ValueError):
        hypoexp_pdf(1.0, [1.0, 1.0], method='closed_form')
    with pytest.raises(ValueError):
        hypoexp_pdf(1.0, [1.0, 1.0], method='decimal')
    with pytest.raises(ValueError):
        hypoexp_pdf(np.linspace(1, 2, 10), [1.0, 2.0], method='convolution')
    with pytest.raises(ValueError):
        hypoexp_cdf(1.0, [1.0, 2.0], method='convolution')
    with pytest.raises(ValueError):
        hypoexp_pdf(1.0, [1.0, 2.0], method='magic')
    with pytest.raises(ValueError):
        Hypoexponential([1.0, -1.0])


def test_weights_and_rvs():
    dist = Hypoexponential([0.5, 1.0, 2.0])
    assert sum(dist.weights) == pytest.approx(1.0)
    samples = hypoexp_rvs([1.0, 2.0, 3.0], size=200000)
    assert samples.shape == (200000,)
    assert samples.mean() == pytest.approx(6.0, rel=0.02)
    assert hypoexp_rvs([1.0, 2.0], size=(3, 4)).shape == (3, 4)
    assert dist.sample(10).shape == (10,)


def test_slow_method_selection_heuristic():
    from hypoexp import _choose_slow_method, _predicted_cost
    two = np.array([1.0, 1.000001])
    twenty = 1.0 + 1e-6 * np.arange(20)
    one_point, many = np.array([1.0]), np.linspace(0.1, 10, 100)
    assert _choose_slow_method(one_point, two, 28, 1e-6) == 'decimal'       # one point, two scales: Decimal is cheaper
    assert _choose_slow_method(many, two, 28, 1e-6) == 'phase_type'         # many points: vectorized phase-type wins
    assert _choose_slow_method(one_point, twenty, 240, 1e-6) == 'phase_type'  # many scales / high precision: Decimal too slow
    assert _choose_slow_method(many, two, 28, 1e-14) == 'decimal'           # more accuracy than doubles allow -> Decimal
    assert _predicted_cost('decimal', 1000, 5, 28) > _predicted_cost('phase_type', 1000, 5, 28, np.linspace(0.1, 10, 1000))


def test_sanity_checks_catch_garbage_and_fall_back(monkeypatch):
    import hypoexp

    def broken_closed_form(x, scales, kind):
        garbage = np.where(np.arange(x.size) % 2 == 0, -1.0, 2.0) * np.exp(-x)  # negative & oscillating
        return garbage, np.zeros(x.size)  # ...while claiming zero rounding error

    monkeypatch.setattr(hypoexp, '_closed_form_method', broken_closed_form)
    x = np.linspace(0.01, 10, 200)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pdf = hypoexp.hypoexp_pdf(x, [1, 2])          # auto: silently recomputed with an exact method
        cdf = hypoexp.hypoexp_cdf(x, [1, 2])
    np.testing.assert_allclose(pdf, np.exp(-x / 2) - np.exp(-x), rtol=1e-12)
    np.testing.assert_allclose(cdf, 1 - 2 * np.exp(-x / 2) + np.exp(-x), rtol=1e-9)  # float reference cancels at small x
    with pytest.warns(UserWarning, match="negative values"):   # forced method: raw output plus a warning
        hypoexp.hypoexp_pdf(x, [1, 2], method='closed_form')


def test_sanity_checks_pass_on_correct_results():
    from hypoexp import _sanity_failures
    x = np.linspace(1e-3, 60, 5000)
    for scales in ([1.0, 2.0], [1.0, 1.0, 3.0], [0.1, 0.1000001]):
        sc = np.array(scales)
        assert _sanity_failures(x, hypoexp_pdf(x, sc), 'pdf', sc, 1e-6) == []
        assert _sanity_failures(x, hypoexp_cdf(x, sc), 'cdf', sc, 1e-6) == []
    assert 'pdf integrates' in _sanity_failures(x, 2 * hypoexp_pdf(x, [1.0, 2.0]), 'pdf', np.array([1.0, 2.0]), 1e-6)[-1]


# ---- phase-type internals: series vs scaling-and-squaring -------------------------------------

import hypoexp as _h


@pytest.mark.parametrize("scales", [[1.0, 2.0], [1.0, 1.0, 2.0], [3.0] * 6, list(1.0 + 1e-6 * np.arange(10)), [0.2, 0.7, 1.5, 4.0]])
@pytest.mark.parametrize("kind", ['pdf', 'cdf'])
def test_phase_type_series_and_squaring_agree(scales, kind):
    scales = np.asarray(scales, dtype=float)
    mu = (1.0 / scales).max()
    x = np.concatenate([[0.0, 1e-9, 1e-3], np.geomspace(0.01, 5e4, 60) / mu])
    a = _h._phase_type_series(x, scales, kind)
    b = _h._phase_type_squaring(x, scales, kind)
    mask = b > 1e-290
    assert np.allclose(a[mask], b[mask], rtol=5e-12, atol=0)
    assert np.all(a[~mask] <= 1e-280)


@pytest.mark.parametrize("scales", [[1.0, 2.0], [0.5, 1.0, 2.0, 4.0, 8.0], list(1.0 + 1e-6 * np.arange(10))])
@pytest.mark.parametrize("kind", ['pdf', 'cdf'])
def test_phase_type_series_matches_reference_up_to_large_lambda(scales, kind):
    scales = np.asarray(scales, dtype=float)
    mu = (1.0 / scales).max()
    x = np.array([1e-6, 0.3, 2.0, 50.0, 700.0, 5e3, 5e4]) / mu
    got = _h._phase_type_series(x, scales, kind) * ((1.0 / scales[-1]) if kind == 'pdf' else 1.0)
    for xi, gi in zip(x, got):
        ref = _reference(xi, scales, kind)
        if ref > 1e-290:
            assert gi == pytest.approx(ref, rel=2e-12), (xi, kind)


def test_phase_type_result_independent_of_series_squaring_split(monkeypatch):
    scales = np.array([0.5, 1.0, 1.0, 3.0])
    x = np.linspace(0, 40, 2001)
    auto = _h._phase_type_method(x, scales, 'pdf')
    monkeypatch.setattr(_h, '_phase_type_plan', lambda lam, n: (np.inf, 0.0))
    all_series = _h._phase_type_method(x, scales, 'pdf')
    monkeypatch.setattr(_h, '_phase_type_plan', lambda lam, n: (-1.0, 0.0))
    all_squaring = _h._phase_type_method(x, scales, 'pdf')
    assert np.allclose(auto, all_series, rtol=1e-12, atol=1e-300)
    assert np.allclose(auto, all_squaring, rtol=1e-12, atol=1e-300)


def test_phase_type_tail_dominated_by_poisson_tail():
    # Erlang(2, 1) at x = 25: c_k is nonzero only at k = 1, far below the Poisson mode k = 25
    assert hypoexp_pdf(25.0, [1.0, 1.0]) == pytest.approx(25 * np.exp(-25), rel=1e-12)
    assert hypoexp_pdf(2000.0, [3.0]) == pytest.approx(np.exp(-2000 / 3) / 3, rel=1e-12)


def test_phase_type_cdf_never_exceeds_one_and_is_fast_for_many_scales():
    scales = np.ones(100)
    x = np.linspace(0, 400, 20001)
    t0 = time.perf_counter()
    c = hypoexp_cdf(x, scales)
    p = hypoexp_pdf(x, scales)
    assert time.perf_counter() - t0 < 5.0
    assert np.all(c <= 1.0) and np.all(np.diff(c) >= -1e-14)  # monotone up to rounding
    assert np.trapezoid(p, x) == pytest.approx(1.0, abs=1e-6)
    assert np.argmax(p) == pytest.approx(np.searchsorted(x, 99.0), abs=5)  # mode of Erlang(100, 1) is 99


# ---- Decimal internals -----------------------------------------------------------------------

def test_decimal_converges_from_too_low_start_precision():
    scales = 1.0 + 1e-6 * np.arange(10)
    x = np.array([0.1, 1.0, 10.0, 30.0])
    value, ok = _h._decimal_method(x, scales, 'pdf', np.full(x.size, 28.0))
    assert ok.all()
    for xi, vi in zip(x, value):
        assert vi == pytest.approx(_reference(xi, scales, 'pdf'), rel=1e-14)


def test_decimal_parallel_matches_serial(monkeypatch):
    scales = np.array([1.0, 1.5, 2.0])
    x = np.linspace(0.1, 10, 64)
    digits = np.full(x.size, 40.0)
    serial, ok_s = _h._decimal_chunk((x, scales, 'pdf', digits, 2000))
    monkeypatch.setattr(_h, '_DECIMAL_PARALLEL_MIN_COST', 0.0)
    parallel, ok_p = _h._decimal_method(x, scales, 'pdf', digits)
    assert ok_s.all() and ok_p.all()
    assert np.array_equal(serial, parallel)


# ---- Randomized stress tests (fixed seeds) -----------------------------------------------------

def _random_scales(rng):
    n = int(rng.choice([1, 2, 3, 5, 8]))
    typ = rng.choice(['spread', 'close', 'wide', 'repeated'])
    if typ == 'spread':
        return rng.uniform(0.1, 5, n)
    if typ == 'close':
        return 1 + 10.0 ** rng.uniform(-9, -3) * np.arange(n)
    if typ == 'wide':
        return 10.0 ** rng.uniform(-3, 3, n)
    return rng.choice(rng.uniform(0.3, 3, max(1, n // 2)), n)


@pytest.mark.parametrize("kind", ['pdf', 'cdf'])
def test_random_cases_within_tolerance_default_and_tight(kind):
    """Default tolerance: every returned value is within 1e-6 relative of a 500-digit reference
    (the float closed form is only accepted when its error bound says so). tolerance=1e-12 forces
    the exact methods, which must agree to ~1e-13. Scalar and array calls must both hold."""
    fn = hypoexp_pdf if kind == 'pdf' else hypoexp_cdf
    rng = np.random.default_rng(2024 if kind == 'pdf' else 2025)
    with warnings.catch_warnings():
        warnings.simplefilter('error')  # no sanity-check failure may be reported on valid input
        for _ in range(25):
            scales = _random_scales(rng)
            xs = 10.0 ** rng.uniform(-6, np.log10(scales.sum() * 8), 5)
            refs = np.array([_reference(x, scales, kind) for x in xs])
            default = fn(xs, scales)
            tight = fn(xs, scales, tolerance=1e-12)
            scalars = np.array([fn(float(x), scales) for x in xs])
            for r, d, t, s in zip(refs, default, tight, scalars):
                if r < 1e-290:
                    continue
                assert d == pytest.approx(r, rel=1e-6)
                assert s == pytest.approx(r, rel=1e-6)
                assert t == pytest.approx(r, rel=1e-12)


def test_random_long_grids_no_false_alarms_and_accurate():
    """Grids of 1000-5000 points (sorted, descending or random spacing, covering or not) must never
    trigger the final sanity checks on valid input and must agree with the exact methods."""
    rng = np.random.default_rng(7)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        for _ in range(20):
            scales = _random_scales(rng)
            mean, sd = scales.sum(), np.sqrt(np.sum(scales ** 2))
            lo = 0.0 if rng.random() < 0.5 else rng.uniform(0, mean)
            hi = rng.uniform(lo + 0.1 * sd, mean + 10 * sd)
            m = int(rng.choice([1000, 2000, 5000]))
            x = np.linspace(lo, hi, m) if rng.random() < 0.7 else np.sort(rng.uniform(lo, hi, m))
            if rng.random() < 0.3:
                x = x[::-1]
            p, c = hypoexp_pdf(x, scales), hypoexp_cdf(x, scales)
            assert np.all(p >= 0) and np.all((c >= 0) & (c <= 1)) and np.all(np.isfinite(p))
            idx = rng.choice(m, 8, replace=False)
            assert np.allclose(p[idx], hypoexp_pdf(x[idx], scales, tolerance=1e-12), rtol=1e-6, atol=1e-6 * p.max())
            assert np.allclose(c[idx], hypoexp_cdf(x[idx], scales, tolerance=1e-12), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("method", ['phase_type', 'decimal'])
@pytest.mark.parametrize("kind", ['pdf', 'cdf'])
def test_forced_exact_methods_on_widely_spread_scales(method, kind):
    scales = np.array([1e-3, 0.02, 0.5, 7.0, 300.0])
    fn = hypoexp_pdf if kind == 'pdf' else hypoexp_cdf
    for x in [1e-6, 1e-3, 0.1, 5.0, 300.0, 3000.0]:
        ref = _reference(x, scales, kind)
        assert fn(x, scales, method=method) == pytest.approx(ref, rel=1e-12, abs=1e-300)


@pytest.mark.parametrize("scales", [[1e-8, 1e8], [1e-8, 1e-8, 1e8], [1e8, 1e-8, 1e-8]])
def test_extreme_scale_ratios(scales):
    # dominated by the slow phase: pdf(1) ~ (1/1e8) e^{-1e-8}, cdf(1) ~ 1e-8 (fast phases ~ finished)
    assert hypoexp_pdf(1.0, scales) == pytest.approx(1e-8 * np.exp(-1e-8), rel=1e-6)
    assert hypoexp_cdf(1.0, scales) == pytest.approx(_reference(1.0, scales, 'cdf'), rel=1e-6)
    assert hypoexp_pdf(1e9, scales) == pytest.approx(1e-8 * np.exp(-10), rel=1e-6)


def test_underflowing_inputs_give_zero_not_nan():
    assert hypoexp_pdf(5e-324, [1e308, 1e308]) == 0.0
    assert hypoexp_cdf(5e-324, [1e308, 1e308]) == 0.0
    assert hypoexp_pdf(1e-320, [1e300, 2e300], method='phase_type') == 0.0
    assert hypoexp_pdf(1e300, [1.0, 2.0]) == 0.0
    assert hypoexp_cdf(1e300, [1.0, 1.0]) == 1.0
    # scale invariance: pdf(cx; c*scales) = pdf(x; scales) / c
    assert hypoexp_pdf(1e-300, [1e-300, 2e-300]) == pytest.approx(hypoexp_pdf(1.0, [1.0, 2.0]) * 1e300, rel=1e-12)
    assert hypoexp_pdf(1e300, [1e300, 2e300]) == pytest.approx(hypoexp_pdf(1.0, [1.0, 2.0]) * 1e-300, rel=1e-12)


def test_many_repeated_and_many_close_scales():
    # Erlang(300, 1) at its mean and 1000 scales 1e-9 apart: deep in Decimal-hostile territory
    assert hypoexp_pdf(300.0, np.ones(300)) == pytest.approx(float(mpmath.exp(-300) * mpmath.mpf(300) ** 299 / mpmath.factorial(299)), rel=1e-11)
    assert hypoexp_cdf(300.0, np.ones(300)) == pytest.approx(float(mpmath.gammainc(300, 0, 300, regularized=True)), rel=1e-11)
    close = 1 + 1e-9 * np.arange(1000)
    assert hypoexp_pdf(1000.0, close) == pytest.approx(float(mpmath.exp(-1000) * mpmath.mpf(1000) ** 999 / mpmath.factorial(999)), rel=1e-6)


def test_convolution_auto_path_matches_exact_and_falls_back_on_coarse_grid():
    rng = np.random.default_rng(11)
    for scales in [np.array([0.3, 1.0, 2.5]), np.array([1.0, 1.0, 0.5]), 1 + 1e-7 * np.arange(4)]:
        x = np.linspace(0, 8 * scales.sum(), 150_001)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            p = hypoexp_pdf(x, scales)
        idx = rng.choice(x.size, 30, replace=False)
        assert np.allclose(p[idx], hypoexp_pdf(x[idx], scales, tolerance=1e-12), atol=1e-6 * p.max())
    with pytest.warns(UserWarning, match='Convolution'):
        p = hypoexp_pdf(np.linspace(0, 1e4, 100_001), [1.0, 2.0])
    assert p[2] == pytest.approx(_reference(0.2, [1.0, 2.0], 'pdf'), rel=1e-6)
