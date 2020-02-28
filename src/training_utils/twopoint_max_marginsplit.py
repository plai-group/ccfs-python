import numpy as np


def twoPointMaxMarginSplit(X, Y, tol):
    """
    This should only be done if X has exactly 2 unique rows in which case it
    produces the optimal split
    """
    bType1 = np.all(np.absolute(np.subtract(X, X[0,:])) < tol, axis=1)
    print('bytpe', bType1)
    YLeft  = Y[bType1,:]
    YRight = Y[~bType1,:]
    print(YLeft)
    print(YRight)

    if np.all(YLeft.sum(axis=0) ==  YRight.sum(axis=0)):
        # Here the two unique points are identical and so we can't helpfully split
        bSp = False
        rmm = []
        cmm = []
        return bSp, rmm, cmm
    else:
        bSp = True;

    # Otherwise the optimal spliting plane is the plane perpendicular
    # to the vector between the two points (rmm) and the maximal
    # marginal split point (cmm) is halway between the two points on this line.
    iType2 = (~bType1).ravel().nonzero()[0][0]
    print('itype')
    print(iType2)
    rmm    = (X[iType2,:] - X[0,:]).T
    print(rmm.shape)
    cmm    = 0.5 * np.add(np.dot(X[iType2,:], rmm), np.dot(X[0,:], rmm))

    if np.any(np.isnan(cmm)) or np.any(np.isinf(cmm)):
        assert (False), 'Suggested split point at infitity or NaN!'

    return bSp, rmm, cmm
