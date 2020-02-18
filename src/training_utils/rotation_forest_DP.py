import numpy as np
from utils.ccfUtils import pcaLite
from utils.commonUtils import manyRandPerms

# TODO: 
def rotationForestDataProcess(X, Y, M, prop_points_subsample, prop_classes_eliminate):
    """
    Carries out the random pca-based projection required for training each
    tree used in a rotation forest
    """
    muX = np.mean(X, axis=0)
    print(muX)
    X   = np.subtract(X, muX)
    print(X)

    D = X.shape[1]

    fOrder  = np.random.permutation(D)
    print(fOrder)

    fGroups = np.reshape(fOrder, (M, M))
    print(fGroups)
    fLeft   = fOrder[(M * np.floor(D/M).astype(int)):]
    print(fLeft)

    K = fGroups.shape[1]

    if prop_classes_eliminate != 0:
        classes = np.unique(Y, axis=0)
        print(classes)
        nClasses = classes.shape[0]

        n_classes_eliminate = np.floor(prop_classes_eliminate * nClasses).astype(int)
        print(n_classes_eliminate)

        classLeaveGroups = manyRandPerms(nClasses, nClasses-n_classes_eliminate, fGroups.shape[1]).T
        print(classLeaveGroups)

        iClasses = {}
        for k in range(nClasses):
            pass

    else:
        classLeaveGroups = np.ones((1,K))
        classLeaveLeft = 1
        iClasses = np.array([0:X.shape[0]])

    R = np.zeros((D,D))
    iUpTo = 0

    # TODO : Complete
    for i in range(K):
        pass


def localRotation(x, p)

    iB = datasample((1:size(x,1)), round(x.shape[0] * p));
    xB = x[iB, :]
    r  = pcaLite(X=x, bScale=False, bMakeFullRank=False)

    return r
