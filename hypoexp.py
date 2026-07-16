import numpy as np
import scipy as sp
from typing import Union
from decimal import Decimal, getcontext
import warnings

def hypoexp_rvs(scales: Union[int, float, list, 'np.ndarray'], size: Union[int, tuple] = 1000) -> Union['np.ndarray', float]:
    """
    Generate random variables from a hypoexponential distribution.

    The hypoexponential distribution represents the sum of independent
    exponential random variables with potentially different rate parameters (scales).

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
    scales_array = np.atleast_1d(np.asarray(scales, dtype=float))

    if np.any(scales_array <= 0):
        raise ValueError("Rate parameters (scales) must be strictly greater than 0.")

    if scales_array.size == 1:
        return sp.stats.expon.rvs(scale=scales_array[0], size=size)

    size_tuple = (size,) if isinstance(size, int) else tuple(size)
    full_size = (scales_array.size,) + size_tuple

    # Reshape scales_array to broadcast across the size_tuple dimensions
    scale_shape = (scales_array.size,) + (1,) * len(size_tuple)
    scales_reshaped = scales_array.reshape(scale_shape)

    return np.sum(sp.stats.expon.rvs(scale=scales_reshaped, size=full_size), axis=0)


def hypoexp_pdf(x: Union[int, float, list, 'np.ndarray'], scales: Union[int, float, list, 'np.ndarray'], tolerance: float = 1e-3) -> Union['np.ndarray', float]:
    """
    Compute the probability density function (PDF) of a hypoexponential distribution.

    The hypoexponential distribution is the distribution of the sum of independent
    exponential random variables with potentially different rate parameters (scales).

    Parameters
    ----------
    x : array_like
        Quantiles, where the PDF will be evaluated.
    scales : array_like
        A single scale parameter or an iterable of scale parameters.
    tolerance : float, optional
        Tolerance for numerical validation (default is 1e-3).

    Returns
    -------
    pdf : ndarray or scalar
        Probability density function evaluated at x.
    """

    ##### General Comments #####
    # There are several ways to compute the PDF of a hypoexponential distribution. See https://en.wikipedia.org/wiki/Hypoexponential_distribution

    x_is_scalar = isinstance(x, (int, float)) or (isinstance(x, np.ndarray) and x.ndim == 0)
    if x_is_scalar:
        x = [float(x)]
        
    if isinstance(scales, (int, float)):
        scales = [float(scales)]
        
    precision = np.float64
    scales = np.array(scales, dtype=precision)  # Ensure scales is a numpy array
    x = np.array(x, dtype=precision)  # Ensure x is a numpy array
    n = len(scales)

    if n == 1:
        res = sp.stats.expon.pdf(x, scale=scales[0])
        return res[0] if x_is_scalar else res


    def closed_form_method(x, scales):
            r"""If the scales are unique, we can use the formula for the pdf of the hypoexponential distribution
            #\sum_i \omega_i^{-1} \exp \left(-x / \omegai\right) \prod{j \neq i} \frac{\omega_i}{\omega_i-\omega_j} where \omega_i = scale[i]#
            this is fast, especially for small len(scales), but due to denominator = scale_i - scale_j growing large if the scale parameters are close,
            there are numerical issues. So, I have implemented a more numerically stable method below that uses the Decimal package. I
            t gets slow for large len(scales), because the required precision to avoid these issues in the worst case grows with len(scales).

            Parameters
            ----------
            x : ndarray
                Quantiles, where the PDF will be evaluated.
            scales : ndarray
                Iterable of scale parameters.

            Returns
            -------
            pdf : ndarray
                Probability density function evaluated at x.
            """

            pdf = np.zeros(len(x), dtype=precision)

            # This is a really neat vectorized implementation of the closed form method. Written in the funnction comment above.
            a = np.tile(scales, (len(scales), 1))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                b = a/(a-a.T)
            b[np.diag_indices(len(scales))] = 1
            prod_contrib = np.prod(b,axis=0)
            exp_pdf = np.array([sp.stats.expon.pdf(x,scale=scales[i]) for i in range(len(scales))])
            pdf = np.dot(prod_contrib,exp_pdf)

            # The code will first try to compute the pdf the "fast" way, and if it fails, it will use the Decimal package to compute the pdf.

            getcontext().prec = len(scales)  # Set baseline precision
            scales_dec = np.vectorize(Decimal)(scales)
            
            # Only verify integration if x has enough points and spans a wide range. 
            # Otherwise, skip the while loop to prevent infinite looping on single points.
            check_integration = len(x) > 1 and (x[-1] - x[0]) > np.sum(scales)
            
            # We check to see if the pdf integrates to 1 within the specified tolerance. If not, we increase the precision and recompute the pdf.
            while check_integration and not np.isclose(sp.integrate.simpson(pdf, x=x), 1, atol=tolerance):
                getcontext().prec += len(scales)
                l_i_at_zero = []
                for i in range(n):
                    product = Decimal(1)
                    scale_i = scales_dec[i]
                    for j in range(n):
                        if i != j:
                            scale_j = scales_dec[j]
                            denominator = scale_i - scale_j
                            if denominator == 0:
                                raise ValueError(f"Duplicate scale parameters detected at indices {i} and {j}.")
                            product *= scale_i / denominator
                    l_i_at_zero.append(product)

                y = np.array([(1/s)*np.exp(-np.vectorize(Decimal)(x)/s) for s in scales_dec])
                pdf = np.vectorize(np.float64)(np.dot(l_i_at_zero,y))

            return pdf
    
    def convolution_method(x, scales):
        """We numerically convolve n exponential distributions. In general, convolving len(scales) exponential distributions is fast,
         but getting accurate results requires good resolution along x, say, len(x) >= 10**5.

        Parameters
        ----------
        x : ndarray
            Quantiles, where the PDF will be evaluated.
        scales : ndarray
            Iterable of scale parameters.

        Returns
        -------
        pdf : ndarray
            Probability density function evaluated at x.
        """
        
        for index, l in enumerate(scales):
            if l <= 0:
                warnings.warn("Rate parameters (lambda) should be greater than 0")
            if index == 0:
                pdf = sp.stats.expon.pdf(x, scale=scales[index])
            else:
                window = sp.stats.expon.pdf(x, scale=scales[index])
                pdf = sp.signal.convolve(pdf, window,mode="full")/np.sum(window)
        # Note: len(pdf)=len(scales)*(len(x)-1)
        return pdf[:len(x)] # Slices the output, so that we return the pdf over the same domain as the input, x

    def phase_type_method(x,scales):
        """This computes the Hypoexponential distribution using the phase-type representation. This involves computing a matrix exponential for a len(scales) X len(scales) matrix,
        multiplied by x, so it scales exponentially with len(x) and worse with len(scales). It is accurate and numerically stable, but slow.
        https://en.wikipedia.org/wiki/Hypoexponential_distribution#Relation_to_the_phase-type_distribution

        Parameters
        ----------
        x : ndarray
            Quantiles, where the PDF will be evaluated.
        scales : ndarray
            Iterable of scale parameters.

        Returns
        -------
        pdf : ndarray
            Probability density function evaluated at x.
        """
        # Note: While MUCH slower than the convolution method, the phase-like representation seems to be more accurate for small values of x.
        
        # # Construct the Theta matrix using the scale parameter s
        Theta = np.zeros((n, n))
        np.fill_diagonal(Theta[:-1, 1:], 1/scales[:-1])
        np.fill_diagonal(Theta, -1/scales)

        # Alpha vector (1, 0, ..., 0)
        alpha = np.zeros(n)
        alpha[0] = 1

        # Vector of ones
        ones_vector = np.ones(n)

        # Perform batch matrix exponentiation for all x values
        exp_x_theta = sp.linalg.expm(np.array([xi * Theta for xi in x]))

        pdf = -np.dot(np.dot(alpha,exp_x_theta), np.dot(Theta, ones_vector))
        
        return pdf
    
    can_integrate = len(x) > 1 and (x[-1] - x[0]) > np.sum(scales)
    
    if not can_integrate and len(x) > 1:
        warnings.warn("The provided domain `x` does not sufficiently cover the distribution (range <= sum of scales). The integration check to verify the PDF will be skipped.")

    if len(x) >= 10**5:
        hypo_pdf =  convolution_method(x, scales)
        if can_integrate and np.isclose(sp.integrate.simpson(hypo_pdf, x=x),1,atol = tolerance):
            return hypo_pdf if not x_is_scalar else hypo_pdf[0]
        elif not can_integrate:
            return hypo_pdf if not x_is_scalar else hypo_pdf[0]
        else:
            if n == len(set(scales)):# if all scales are unique
                res = closed_form_method(x, scales) # closed form method guarantees that the pdf integrates to 1 within tolerance.
                return res if not x_is_scalar else res[0]
            else:
                hypo_pdf = phase_type_method(x, scales)
                if can_integrate and not np.isclose(sp.integrate.simpson(hypo_pdf, x=x),1,atol = tolerance):
                    print("Warning: The PDF does not integrate to 1 within tolerance. The pdf's integral is: ",sp.integrate.simpson(hypo_pdf, x=x))
                return hypo_pdf if not x_is_scalar else hypo_pdf[0]
    else:
        if n == len(set(scales)):# if all scales are unique
            res = closed_form_method(x, scales) # closed form method guarantees that the pdf integrates to 1 within tolerance.
            return res if not x_is_scalar else res[0]
        else:
            hypo_pdf = phase_type_method(x, scales)
            if can_integrate and not np.isclose(sp.integrate.simpson(hypo_pdf, x=x),1,atol = tolerance):
                print("Warning: The PDF does not integrate to 1 within tolerance. The pdf's integral is: ",sp.integrate.simpson(hypo_pdf, x=x))
            return hypo_pdf if not x_is_scalar else hypo_pdf[0]


class Hypoexponential:
    """
    Object-oriented interface for the Hypoexponential distribution.
    This class wraps the generalized functions for computing PDF and sampling.
    """
    # only works if the parameters eta are distinct for CDF currently
    def __init__(self, eta):
        self._eta = eta
        self._scales = [1.0 / e for e in eta]
        self._prod_eta = []
        self._weights = []
        for i, eta_i in enumerate(self._eta):
            tmp_list = list(self._eta[:i])
            tmp_list.extend(self._eta[i+1:])
            self._prod_eta.append(np.prod([(eta_j - eta_i) for eta_j in tmp_list]))
            self._weights.append(np.prod([1 / (1 - eta_i/eta_j) for eta_j in tmp_list]))

    def pdf(self, x):
        return hypoexp_pdf(x, self._scales)

    def cdf(self, x):
        return [sum([(1-np.exp(-eta_j * xx)) * self._weights[j] for j, eta_j in enumerate(self._eta)]) for xx in x]

    @property
    def weights(self):
        return self._weights

    @property
    def params(self):
        return {'rates': self._eta}

    def sample(self, n_sample=1):
        return hypoexp_rvs(self._scales, size=n_sample)


if __name__ == '__main__':
    from numpy import linspace, array
    import matplotlib.pyplot as plt

    eta = [0.05665075, 0.07182203, 0.05739665]
    eta = list(set(eta))

    sum_exp = Hypoexponential(eta)
    sample = sum_exp.sample(1000)
    t_range = linspace(0.001, max(sample))
    plt.hist(sample, bins=50, density=True)
    plt.plot(t_range, sum_exp.pdf(t_range))
    plt.show()

    print(sum_exp.params)
    print(sum(sum_exp.weights))
    plt.plot(t_range, sum_exp.cdf(t_range))
    plt.show()
