from mxnet import random
from mxnet.initializer import Initializer


class TruncNorm(Initializer):
    r"""Initialize the weight by drawing sample from truncated normal distribution with
    provided mean and standard deviation. Values whose magnitude is more than 2 standard deviations
    from the mean are dropped and re-picked..
    Parameters
    ----------
    mean : float, default 0
        Mean of the underlying normal distribution
    stdev : float, default 0.01
        Standard deviation of the underlying normal distribution
    **kwargs : dict
        Additional parameters for base Initializer.
    """
    def __init__(self, mean=0, stdev=0.01, **kwargs):
        super(TruncNorm, self).__init__(**kwargs)
        try:
            from scipy.stats import truncnorm
        except ImportError:
            raise ImportError('SciPy is not installed. '
                              'You must install SciPy >= 1.0.0 in order to use the '
                              'TruncNorm. You can refer to the official '
                              'installation guide in https://www.scipy.org/install.html .')

        self._frozen_rv = truncnorm(-2, 2, mean, stdev)

    def _init_weight(self, name, arr):
        # pylint: disable=unused-argument
        """Abstract method to Initialize weight."""
        arr[:] = self._frozen_rv.rvs(arr.size).reshape(arr.shape)


class NormalScaled(Initializer):
    """Initializes weights with random values sampled from a normal distribution
    with a mean of zero and standard deviation of `sigma`.

    Parameters
    ----------
    sigma : float, optional
        Standard deviation of the normal distribution.
        Default standard deviation is 0.01.
    """
    def __init__(self, scale, sigma=0.01):
        super(NormalScaled, self).__init__(sigma=sigma)
        self.sigma = sigma
        self.scale = scale

    def _init_weight(self, _, arr):
        random.normal(0, self.sigma, out=arr)
        arr[:] = arr[:] / self.scale
