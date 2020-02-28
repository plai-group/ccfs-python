import numpy as np

def cohenKappa(confusionMatrix):
    """
    Calculates Cohen's kappa from a confusion matrix.

    Parameters
    ----------
    confusionMatrix: Numpy array

    Returns
    -------
    kappa: float
    """
    propPreds = np.sum(confusionMatrix, axis=0)
    propPreds = propPreds/np.sum(propPreds)

    propReal  = np.sum(confusionMatrix, axis=1)
    propReal  = propReal/np.sum(propReal)

    p_e = np.sum(propPreds * propReal)
    p_o = np.sum(np.diag(confusionMatrix))/np.sum(confusionMatrix)

    kappa = (p_o - p_e)/(1 - p_e)

    return kappa


def f1score(ypred, ytrue):
    """
    Calculates F1 Score between prediction and ground-truth

    Parameters
    ----------
    ypred: Numpy array
    ytrue: Numpy array

    Returns
    -------
    f1_raw: Numpy array
    f1_weighted_mean: float
    """
    def my_equal(a, b):
        return np.equal(b, a, dtype=int)

    preds = np.unique(ypred)
    trues = np.unique(ytrue)
    vals  = np.union1d(preds,trues)[np.newaxis] # (n x 1)

    b_pred = np.apply_along_axis(my_equal, axis=1, arr=vals.T, b=ypred)
    b_true = np.apply_along_axis(my_equal, axis=1, arr=vals.T, b=ytrue)

    tp = np.sum(np.logical_and(b_pred, b_true), axis=0)
    all_true = np.sum(b_true, axis=0);
    all_pred = np.sum(b_pred, axis=0);

    f1_raw = 2 * tp / (all_true + all_pred);
    f1_weighted_mean = np.sum(f1_raw * all_true / (np.sum(all_true)))

    return f1_weighted_mean, f1_raw


def fastUnique(X):
    """
    Unique values in array

    Parameters
    ----------
    X: Numpy Vector array

    Returns
    -------
    uX: Numpy array
    """
    is_row = False # Check Row or Column vector
    if len(X.shape) == 1:
        uX = np.sort(X) # Row Vector
        is_row = True
    else:
        uX = np.sort(X, axis=0) # Column Vector

    if is_row:
        uX = uX[np.concatenate((np.array([True]), np.diff(uX, n=1, axis=0) !=0), axis=0)]
    else:
        uX = uX[np.concatenate((np.array([[True]]), np.diff(uX, n=1, axis=0) !=0), axis=0)][np.newaxis].T

    return uX


def sVT(X):
    """
    Transpose numpy array for single dimension array.

    Parameters
    ----------
    X: Numpy Vector array

    Returns
    -------
    sX: Vector array
    """
    if len(X.shape) == 1:
        sX = X[np.newaxis]
        return sX.T
    else:
        sX = X.T

        return sX


def queryIfColumnsVary(X, tol):
    """
    Function that says whether columns are constant or not

    Parameters
    ----------
    X: Numpy array
    tol: Float

    Returns
    -------
    bVar: Numpy Boolean array
    """
    bVar = np.max(np.absolute(np.diff(X[0:min(5, X.shape[0]), :], axis=0)), axis=0) > tol
    bVar[~bVar] = np.max(np.absolute(np.diff(X[:, ~bVar], axis=0)), axis=0) > tol

    return bVar


def queryIfOnlyTwoUniqueRows(X):
    """
    Function that checks if an array has only two unique rows as this can
    cause failure in case of LDA.

    Parameters
    ----------
    X: Numpy array

    Returns
    -------
    bVar: Numpy Boolean array
    """
    # def my_equal(a, b):
    #     return np.equal(b, a, dtype=int, casting="unsafe")

    if X.shape[0] == 2:
        bLessThanTwoUniqueRows = True
        return bLessThanTwoUniqueRows

    #eqX = np.apply_along_axis(my_equal, axis=1, arr=X, b=X[0,:])

    eqX = np.equal(X, X[0, :])
    bEqualFirst = np.all(eqX, axis=1)

    iFirstNotEqual = (~bEqualFirst).ravel().nonzero()[0]
    if iFirstNotEqual.size == 0:
        bLessThanTwoUniqueRows = True
        return bLessThanTwoUniqueRows

    iToCheck = ((~bEqualFirst[1:]).ravel().nonzero()[0]) + 1
    # Xit = np.apply_along_axis(my_equal, axis=1, arr=X[iToCheck, :], b=X[iFirstNotEqual[0], :])
    Xit = np.equal(X[iToCheck, :], X[iFirstNotEqual[0], :])
    bNotUnique = np.all(Xit, axis=1)
    bLessThanTwoUniqueRows = np.all(bNotUnique, axis=0)

    return bLessThanTwoUniqueRows


def zScoreToX(zScore, mu_X, std_X):
    """
    Convert Score back to X

    Parameters
    ----------
    zScore: Numpy array
    tol: Float

    Returns
    -------
    X: Numpy array
    """
    X = np.add(zScore * std_X, mu_X)
    X[np.isnan(X)] = 0

    return X


def is_numeric(X, compress=True):
    """
    Determine whether input is numeric array

    Parameters
    ----------
    X: Numpy array
    compress: Boolean

    Returns
    -------
    V: Numpy Boolean array if compress is False, otherwise Boolean Value
    """
    def is_float(val):
        try:
            float(val)
        except ValueError:
            return False
        else:
            return True

    isnumeric = np.vectorize(is_float, otypes=[bool]) # return numpy array
    V = isnumeric(X)

    if compress:
        return np.all(V)

    return V


def makeSureString(A, nSigFigTol, access_all = False):
    """
    Ensure that all numerical values are strings

    Parameters
    ----------
    A: Numpy array
    nSigFigTol: Float

    Returns
    -------
    A: Numpy array
    """
    # To Match MATLAB function
    def num2str(val):
        if val <= 9:
            try:
                rval = ord(str(val))
            except ValueError or TypeError:
                return val
            else:
                return rval
        else:
            return val

    if type(A) == int or type(A) == float:
        bNum = is_numeric(A)
        if bNum:
            A = num2str(A)
    else:
        if A.size > 1 and access_all:
            num2str_f = np.vectorize(num2str, otypes=[float]) # return numpy array
            bNum = is_numeric(A)
            if bNum:
                A = num2str_f(A)
        elif A.size > 1:
            bNum = is_numeric(A)
            if bNum:
                A[0,0] = num2str(A[0,0])
        else:
            bNum = is_numeric(A)
            if bNum:
                A = num2str(A)

    return A


def dict2array(X):
    """
    Returns a Numpy array from dictionary

    Parameters
    ----------
    X: dict
    """
    all_var = []

    for k in X.keys():
        all_var.append(X[k])

    return np.array(all_var)


def amerge(a, b, p):
    """
    Advanced merging for Numpy array similar to MATLAB
    """
    a1, a2 = a.shape
    b1, b2 = b.shape
    p1 = p.shape[0]

    if p1 > a1:
        arr = np.zeros((p1, b2))
        for k in p:
            arr[k, :] = b[k]

        return arr
    else:
        a[p, :] = b

        return a

def islogical(X):
    """
    Check if the Numpy array is a valid Boolean array
    """
    def isBool(val):
        if int(val) == 1 or int(val) == 0:
            return True
        else:
            return False

    isBool_f = np.vectorize(isBool, otypes=[bool]) # return numpy array
    all_vals = isBool_f(X)

    return np.all(all_vals)
