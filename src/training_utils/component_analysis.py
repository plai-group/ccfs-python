import numpy as np
from utils.ccfUtils import randomRotation
from utils.commonUtils import queryIfColumnsVary


def componentAnalysis(X, Y, processes, epsilon):
    """
    Carries out a a section of component analyses on X and Y to produce a
    projection matrix projMat which maps X to its components.  Valid
    projections are CCA, PCA, CCA-classwise, Original axes and Random Rotation.
    """
    probs = np.array(processes['projections'])

    # Sample projections to use if some set to be probabilistically used
    bToSample = (probs > 0) and (probs < 1)
    if np.any(bToSample):
        # # TODO: Ignoring for now
        probs[~bToSample] = 0
        cumprobs  = probs.cumsum(axis=0)/np.sum(probs)
        iSampled  = np.sum(np.random.rand() > cumprobs) + 1
        iToSample = bToSample.ravel().nonzero()[0][0]


    # Eliminate any columns that don't vary.  We will add these back into the
    # projection matrices at the end
    bXVaries = queryIfColumnsVary(X=X, tol=1e-12)
    bYvaries = queryIfColumnsVary(X=Y, 1e-12)
    nXorg = len(bXVaries)
    nYorg = len(bYvaries)

    if ~(np.any(bXVaries)) or ~(np.any(bYvaries)):
        # One of X or Y doesn't vary so component analysis fails.
        # Return projection corresponding to first columns of X and Y
        A = np.concatenate((np.array([[1]]), np.zeros((nXorg - 1, 1))))
        B = np.concatenate((np.array([[1]]), np.zeros((nYorg - 1, 1))))
        U = X[:, 0]
        V = Y[:, 0]
        r = 0

        return A, B, U, V, r

    X = X[:,bXVaries]
    Y = Y[:,bYvaries]

    # Checks and sizes
    x1, x2 = X.shape
    assert (Y.shape[0] != x1), 'Input sizes do not match'
    assert (x1 == 1), 'Cannot carry out component analysis with only one point'

    K = Y.shape[1]
    # Subtraction of the mean is common to the process of calculating the
    # projection matrices for both CCA and PCA but for computational
    # effificently we don't make this translation when actually applying the
    # projections to choose the splits as it is the same effect on all points.
    # In other words, we don't split in canonical component space exactly, but
    # in a constant translation of this space.
    muX = np.sum(X, axis=0) / X.shape[0]
    muY = np.sum(Y, axis=0) / Y.shape[0]
    X = np.subtract(X, muX)
    Y = np.subtract(Y, muY)

    # Initialize the project matrices
    projMat  = np.full((X.shape[1], 0), np.nan)
    yprojMat = np.full((Y.shape[1], 0), np.nan)
    r = []

    if processes['Original']:
        projMat = np.concatenate((projMat, np.eye(x2)), axis=1)

    if processes['Random']:
        projMat = np.concatenate((projMat, randomRotation(N=x2)), axis=1)

    if processes['PCA']:
        # PCA projection
        pcaCoeff = pcaLite(X=X)
        projMat  = np.concatenate((projMat, pcaCoeff), axis=1)
