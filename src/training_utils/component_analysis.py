import numpy as np
import scipy.linalg as la
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

    if processes['CCA'] or processes['CCAclasswise']:
        # CCA based projections
        q1, r1, p1 = la.qr(X, pivoting=True)
        # Reduce to full rank within some tolerance
        if r1.size == 0:
            rankX = 0
        else:
            rankX = np.sum(np.abs(np.diag(r1)) >= (epsilon * np.abs(r1[0])))

        if rankX == 0:
            A = np.concatenate((np.array([[1]]), np.zeros((nXorg - 1, 1))))
            B = np.concatenate((np.array([[1]]), np.zeros((nYorg - 1, 1))))
            U = X[:, 0]
            V = Y[:, 0]
            r = 0

            return A, B, U, V, r

        elif rankX < x2:
            q1 = q1[:, 0:rankX]
            r1 = r1[0:rankX, 0:rankX]

        if processes['CCA']:
            q2, r2, p2 = la.qr(Y, pivoting=True)
            # Reduce to full rank within some tolerance
            if r2.size == 0:
                rankY = 0
            else:
                rankY = np.sum(np.abs(np.diag(r2)) >= (epsilon * np.abs(r2[0])))

            if rankY == 0:
                A = np.concatenate((np.array([[1]]), np.zeros((nXorg - 1, 1))))
                B = np.concatenate((np.array([[1]]), np.zeros((nYorg - 1, 1))))
                U = X[:, 0]
                V = Y[:, 0]
                r = 0

                return A, B, U, V, r

            elif rankY < K:
                q2 = q2[:, 0:rankY]

            # Solve CCA using the decompositions, taking care to use minimal
            # complexity orientation for SVD.  Note the two calculations are
            # equivalent except in computational complexity
            d = min(rankX, rankY)

            if rankX >= rankY:
                L, D, M = np.linalg.svd(q1.T @ q2)
            else:
                M, D, L = np.linalg.svd(q2.T @ q1)

            locProj = linalg.solve(r1, L[:, 0:d] * np.sqrt(x1 - 1))
            # Put coefficients back to their full size and their correct order
            locProj[p1, :] = np.concatenate((locProj, np.zeros((x2-rankX, d)))) # Maybe fix with axis
            projMat = np.concatenate((projMat, locProj)) # Maybe fix with axis

            r2 = r2[0:rankY, 0:rankY]
            locyProj = linalg.solve(r2, M[:, 0:d] * np.sqrt(x1-1))
            locyProj[p2, :] = np.concatenate((locyProj, np.zeros((K-rankY, d))))
            yprojMat = np.concatenate((yprojMat,locyProj)) # Maybe fix with axis

            r = np.minimum(np.maximum(np.diag(D[:,1:d]).T, 0), 1)

        if processes['CCAclasswise']:
            # Consider each output in an in / out fashion to generate a set of K projections.
            for k in range(K):
                L, _, _ = np.linalg.svd(q1.T @ Y[:, k])
                locProj = linalg.solve(r1, L[:, 0] * np.sqrt(x1-1))
                locProj[p1, :] = np.concatenate((locProj, np.zeros((x2-rankX,1))))
                projMat = np.concatenate((projMat,locProj))

    # Normalize the projection matrices.  This ensures that the later tests for
    # close points are triggered appropriately and is useful for interpretability.
    projMat = np.divide(projMat, np.sqrt(np.sum(projMat**2, axis=0)))

    # Note that as in general only a projection matrix is given, we need to
    # add the mean back to be consistent with general use.  This equates to
    # addition of a constant term to each column in U
    U = X @ projMat
    V = Y @ yprojMat

    # Finally, add back in the empty rows in the projection matrix for the
    # things which didn't vary
    A = np.zeros((nXorg, projMat.shape[1]))
    A[bXVaries, :] = projMat
    B = np.zeros((nYorg, yprojMat.shape[1]))
    B[bYvaries, :] = yprojMat

    return A, B, U, V, r