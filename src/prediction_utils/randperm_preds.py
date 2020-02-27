from utils.ccfUtils import random_missing_vals


def randperm_preds(tree, X, bOutOfBag=None):
    """
    Calculates D sets of predictions for a tree, each with column d of X
    randomly permuted. Currently only used by feature_importance function.

    Parameters
    ----------
    tree      = tree
    X         = Samples to test
    bOutOfBag = Use only the out of bag indices, requires CCF-Bag to
                have been used in the first place.

    Returns
    -------
    YpermPreds = predicts
    """

    if bOutOfBag
        bOutOfBag = True

    if bOutOfBag:
        X = X[tree["iOutOfBag"], :]

    # Any values left as NaN now need to be randomly assigned
    X = random_missing_vals(X)
    N, D = X.shape
    YpermPreds = {}

    for d in range(D):
        Xd_true = X[:, ]
        X[:, d] = X[np.random.permutation(N), d]
        YpermPreds[d] = predictFromCCT(tree, X)
        X[:, d] = Xd_true

    YpermPreds[D] = predictFromCCT(tree,X)

    return YpermPreds
