""" py.test test for glmtools code

Run with:

    py.test glmtools
"""

import numpy as np
import scipy.stats


from glmtools import glm, t_test

from numpy.testing import assert_almost_equal


def test_glm_t_test():
    # Test glm and t_test against scipy
    # Your test code here

    # Testing obvious things
    x = np.ones((10, 1))
    y = np.ones((10, 1)) * 2
    b_test, sigma_2_test, df_test = glm(y, x)
    assert(np.allclose(b_test, [2]))
    assert(df_test == 9)
    assert(sigma_2_test == 0)


    # Testing against scipy
    n_test = 10
    c = np.array([0, 1])

    for i in range(n_test):
        x = np.random.normal(10, 2, size=20)
        y = np.random.normal(20, 1, size=20)

        X = np.ones((20, 2))
        X[:, 1] = x

        b_test, sigma_2_test, df_test = glm(y, X)
        tval_test, pval_test = t_test(c, X, b_test, sigma_2_test, df_test)

        res = scipy.stats.linregress(x, y)

        assert(np.allclose(b_test, [res.intercept, res.slope]))
        assert(np.isclose(pval_test, res.pvalue))

    return
