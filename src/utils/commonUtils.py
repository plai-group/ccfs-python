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
    bVar = np.max(np.abs(np.diff(X[0:min(5, X.shape[0]), :], axis=0)), axis=0) > tol
    bVar[~bVar] = np.max(np.abs(np.diff(X[:, ~bVar], axis=0)), axis=0) > tol

    return bVar


def zScoreToX(zScore, mu_X, std_X):
    """
    Convert Score to orginal X

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
