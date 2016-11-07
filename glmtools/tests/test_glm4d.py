""" Test glm4d module

Run with:

    py.test glmtools
"""

import numpy as np
import scipy.stats

from glmtools import glm_4d, t_test_3d, glm, t_test

from numpy.testing import assert_almost_equal, assert_equal


def test_glm4d():
    # Test GLM model on 4D data
    # +++your code here+++

    # Testing the obvious
    X = np.ones((10, 1))
    Y = np.ones((3, 3, 3, 10)) * 2
    B, sigma_2, df = glm_4d(Y, X)
    assert(np.allclose(sigma_2, np.zeros(sigma_2.shape)))
    assert(df == 9)
    assert(np.allclose(B, np.ones(B.shape) * 2))

    # Test against scipy
    n_test = 10
    c = np.array([0, 1])

    for i in range(n_test):
        x = np.random.normal(10, 2, size=20)
        X = np.ones((20, 2))
        X[:, 1] = x

        y = np.random.rand(10, 10, 5, 20)
        Y = y.reshape((-1, y.shape[-1]))
        Y = Y.T

        # Results from own functions
        b_test, sigma_2_test, df_test = glm_4d(y, X)
        tval_test, pval_test = t_test_3d(c, X, b_test, sigma_2_test, df_test)

        # Get corresponding values from scipy.stats
        res_b = np.zeros(b_test.shape)
        res_p = np.zeros(b_test.shape[:-1])
        for i in range(10):
            for j in range(10):
                for k in range(5):
                    r = scipy.stats.linregress(x, y[i, j, k, :])
                    res_b[i, j, k, :] = [r.intercept, r.slope]
                    res_p[i, j, k] = r.pvalue

        res_p = res_p.reshape(pval_test.shape)

        assert(np.allclose(b_test, res_b))
        assert(np.allclose(pval_test, res_p))
    return
