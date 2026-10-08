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
        assert func(xi, scales) == gi  # scalar call gives the same value as the array call


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
    for grid in (np.array([2.0]), np.linspace(0, 2, 3), np.linspace(1.5, 2, 100000)[::-1], np.array([[2.0, 7.0]])):
        assert hypoexp_pdf(grid, [1.0, 1.00000001, 1.00000002]).flat[0] == pytest.approx(expected, rel=1e-12)


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
