import numpy as np
from utils.commonUtils import sVT
from utils.ccfUtils import pcaLite
from utils.ccfUtils import manyRandPerms

def localRotation(x, p):
    iB = np.random.choice((np.arange(0, x.shape[0])), round(x.shape[0] * p))
    xB = x[iB, :]
    r, _, _  = pcaLite(X=x, bScale=False, bMakeFullRank=False)

    return r


def rotationForestDataProcess(X, Y, M, prop_points_subsample, prop_classes_eliminate):
    """
    Carries out the random pca-based projection required for training each
    tree used in a rotation forest
    """
    muX = np.mean(X, axis=0)
    X   = np.subtract(X, muX)

    D = X.shape[1]

    fOrder  = np.random.permutation(D)
    fGroups = np.reshape(fOrder[0:int(M * np.floor(D/M))], (M, -1), order='F')
    fLeft   = fOrder[int(M * np.floor(D/M) + 1):]
    print('m \n', fGroups)

    K = fGroups.shape[1]

    if prop_classes_eliminate != 0:
        classes = np.unique(Y, axis=0)
        print(classes)
        nClasses = classes.shape[0]
        print(nClasses)

        n_classes_eliminate = np.floor(prop_classes_eliminate * nClasses).astype(int)
        print(n_classes_eliminate)

        classLeaveGroups = manyRandPerms(nClasses, nClasses-n_classes_eliminate, fGroups.shape[1]).T
        print(classLeaveGroups)
        classLeaveLeft   = np.random.choice(nClasses, nClasses-n_classes_eliminate)
        print(classLeaveLeft)

        iClasses = []
        for k in range(nClasses):
            iClasses.append((((Y[:, k]).ravel().nonzero()[0])[np.newaxis]))
        iClasses = np.array(iClasses)
        print(iClasses)
        print(iClasses.shape)
    else:
        classLeaveGroups = np.ones((1,K))
        classLeaveLeft   = 1
        iClasses_x       = np.arange(0, X.shape[0])
        iClasses         = np.array(iClasses_x)

    R = np.zeros((D,D))
    iUpTo = 0

    for n in range(K):
        cLGidx = list(classLeaveGroups[:, n])
        iThis  = iClasses[cLGidx]
        r = localRotation(x=X[iThis, fGroups[:, n]], p=prop_points_subsample)
        print(r)
        print(r.shape)
        print(((1 + (n-1)*M)))
        print((n*M))
        print((iUpTo))
        print((iUpTo + r.shape[1] - 1))

        R[((1 + (n-1)*M)):(n*M), (iUpTo):(iUpTo + r.shape[1] - 1)] = r
        iUpTo = iUpTo + r.shape[1]

    if not(fLeft.size == 0):
        iThis = iClasses[classLeaveLeft.flatten()]
        r = localRotation(x=X[iThis, fLeft], p=prop_points_subsample)
        R[((1+(K)*M)-1):, (iUpTo-1):(iUpTo + r.shape[1] - 1)] = r

    R[fOrder, :] = R
    U = X @ R

    return R, muX, U
