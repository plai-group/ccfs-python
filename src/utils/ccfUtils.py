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
        rand_range = np.arange(0, nVarTot) # Changed to match Python Idxing
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
         X   = np.divide(X, sig)

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


def randomRotation(N):
    """
    Random rotation matrix of given dimension

    Parameters
    ----------
    N: int

    Returns
    -------
    Q: Numpy array
    """
    Q, R = np.linalg.qr(np.random.randn(N, N))
    Q = Q @ np.diag(np.sign(np.diag(R)))

    detR = np.linalg.det(Q)
    if np.round(detR) == -1:
        Q[:, 0] = -Q[:, 0]

    return Q


def regCCA_alt(X, Y, gammaX, gammaY, corrTol):
    """
    Fast regularized CCA.  Used when doing kernel CCA.

    Parameters
    ----------
    X: Numpy array
    Y: Numpy array
    gammaX: float
    gammaY: float
    corrTol: float

    Returns
    -------
    A: Numpy array
    """
    D = X.shape[1]
    K = Y.shape[1]

    XY = np.concatenate((X, Y), axis=1)
    C = np.cov(XY, rowvar=False) # rowvar=False to match MATLAB

    Cxx = C[0:D, 0:D] + (gammaX * np.eye(D))
    Cyy = C[D:, D:] + (gammaY * np.eye(K))
    Cxy = C[0:D, D:]

    Cxx = 0.5 * (Cxx + Cxx.T);
    Cyy = 0.5 * (Cyy + Cyy.T);

    CholCxx = np.linalg.cholesky(Cxx).T
    if CholCxx.shape[0] == CholCxx.shape[1]:
        invCholCxx = np.linalg.solve(CholCxx, np.eye(D))
    else:
        invCholCxx = np.linalg.lstsq(CholCxx, np.eye(D))

    CholCyy = np.linalg.cholesky(Cyy).T
    if CholCyy.shape[0] == CholCyy.shape[1]:
        invCholCyy = np.linalg.solve(CholCyy, np.eye(K))
    else:
        invCholCyy = np.linalg.lstsq(CholCyy, np.eye(K))

    T = invCholCxx.T @ Cxy @ invCholCyy

    if D >= K:
        [L,S,D] = np.linalg.svd(T, 0)
        r = np.diag(S)
        A = invCholCxx @ L
        B = invCholCyy @ D
    else:
        [L,S,D] = np.linalg.svd(T.T, 0);
        r = np.diag(S)
        A = invCholCxx @ D
        B = invCholCyy @ L

    bGreaterThanTol = np.absolute(r) > np.absolute(corrTol * np.max(np.absolute(r)))

    A = A[:, bGreaterThanTol[0]]
    B = B[:, bGreaterThanTol[0]]
    r = r[bGreaterThanTol]

    return A, B, r
