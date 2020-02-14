import numpy as np

def genFeatureExpansionParameters(X, nF, s=0.1):
    """
    Generates random feature parameters for kernel CCA.

    Parameters
    ----------
    X: Numpy array
    nF: float
    S: float

    Returns
    -------
    w: Numpy array
    b: Numpy array
    """
    M = X.shape[1]

    w = s * np.random.randn(M, nF)
    b = 2 * np.pi *np.random.rand(1, nF)

    return w, b


def manyRandPerms(nVarTot, nVarSel, nTimes):
    """
    Generate a number of random permutations.

    Parameters
    ----------
    nVarTot: int
    nVarSel: int
    nTimes: int

    Returns
    -------
    terms: Numpy array
    """
    terms = np.empty((nTimes, nVarSel))
    terms.fill(np.nan)

    for n in range(nTimes):
         # TODO: change maybe depending on idx
        rand_range = np.arange(1, nVarTot+1)
        terms[n,:] = np.random.permutation(rand_range)

    return terms


def pcaLite(X, bScale=False, bMakeFullRank=True):
    """
    A faster and more natural PCA function.
     - If bScale is true then the data is scaled according to the individual
       feature variances. Default = False.
     - If bMakeFullRank is true then the principle components are reduced to
       ensure that they are full rank. Default = True.

    Parameters
    ----------
    X: Numpy array
    bScale: Boolean
    bMakeFullRank: Boolean

    Returns
    -------
    coeff: Numpy array
    muX: Numpy array
    vals: Numpy array
    """
    eps = 2.2204e-16
    muX = np.mean(X, axis=0)

    X = np.subtract(X, muX)
    if bScale:
         sig = np.std(X, axis=0, ddof=1) # DDof set to 1 to match MATLAB std
         X = np.divide(X, sig)

    _, v, coeff = np.linalg.svd(X)

    if bMakeFullRank:
        coeff = coeff[:, v > (eps * v[0] * max(X.shape))]

    vals = X @ coeff # Matmul

    if bScale:
        sig = sig[np.newaxis]
        coeff = np.divide(coeff, sig.T)

    return coeff, muX, vals


def random_feature_expansion(X, w, b):
    """
    Used to expand the random features in kernel CCA

    Parameters
    ----------
    X: Numpy array
    w: Numpy array
    b: Numpy array

    Returns
    -------
    Z: Numpy array
    """
    Z = np.cos(np.add(X@w, b))

    return Z


def random_missing_vals(X, mu=0, sig=1):
    """
    Randomly assigns missing values to draws from a normal with the mean and
    standard deviation of the data.

    Parameters
    ----------
    X: Numpy array
    mu: float
    sig: float

    Returns
    -------
    X: Numpy array
    """
    bNaN   = np.isnan(X)
    nRands = np.sum(bNaN[:])

    if nRands != 0:
        X[bNaN] = sig * np.random.randn(nRands) + mu

    return X
