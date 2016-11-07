""" Run GLM on final dimension of 4D arrays
"""

import numpy as np
import numpy.linalg as npl
import scipy.stats

from .glm import glm, t_test


def glm_4d(Y, X):
    """ Run GLM on on 4D data `Y` and design `X`

    Parameters
    ----------
    Y : array shape (I, J, K, T)
        4D array to fit to model with design `X`.  Column vectors are vectors
        over the final length T dimension.
    X : array ahape (T, P)
        2D design matrix to fit to data `Y`.

    Returns
    -------
    B : array shape (I, J, K, P)
        parameter array, one length P vector of parameters for each voxel.
    sigma_2 : array shape (I, J, K)
        unbiased estimate of variance for each voxel.
    df : int
        degrees of freedom due to error.
    """
    # +++your code here+++
    Y_2d = Y.reshape((-1, Y.shape[-1])).T
    X_inv = npl.pinv(X)

    B = X_inv.dot(Y_2d)
    B_reshape = B.T.reshape((Y.shape[0], Y.shape[1], Y.shape[2], X.shape[1]))

    df = X.shape[0] - npl.matrix_rank(X)

    e = Y_2d - X.dot(B)
    sigma_2 = np.sum(e ** 2, axis=0) / df
    sigma_2 = sigma_2.reshape(Y.shape[:-1])

    return B_reshape, sigma_2, df


def t_test_3d(c, X, B, sigma_2, df):
    """ Two-tailed t-test on 3D estimates given contrast `c`, design `X`

    Parameters
    ----------
    c : array shape (P,)
        contrast specifying conbination of parameters to test.
    X : array shape (N, P)
        design matrix.
    B : array shape (I, J, K, P)
        parameter array, one length P vector of parameters for each voxel.
    sigma_2 : array shape (I, J, K)
        unbiased estimate of variance for each voxel.
    df : int
        degrees of freedom due to error.

    Returns
    -------
    t : array shape (I, J, K)
        t statistics for each data vector.
    p : array shape (V,)
        two-tailed probability value for each t statistic.
    """
    # Your code code here

    B_2d = B.reshape((-1, B.shape[-1]))

    c_b_cov = c.dot(npl.pinv(X.T.dot(X))).dot(c)

    t = c.dot(B_2d.T) / np.sqrt(sigma_2.ravel() * c_b_cov)

    t_dist = scipy.stats.t(df=df)

    p_value = np.zeros(np.prod(B.shape[:-1]))

    # Calculate t values
    for i in range(len(t)):
        if t[i] > 0:
            p_value[i] = 2 * (1 - t_dist.cdf(t[i]))
        else:
            p_value[i] = 2 * t_dist.cdf(t[i])

    t = t.reshape(B.shape[:-1])

    return t, p_value
