import sys
sys.path.append("/ccfs-python/src/")

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from predict_from_CCF import predictFromCCF
from utils.commonUtils import islogical
from utils.ccfUtils import mat_unique

def plotCCFDecisionSurface(name, CCF, x1Lims, x2Lims, XTrain, X, Y, nx1Res=250, nx2Res=250, n_contours_or_vals=[], plot_X=True):
    xi = np.linspace(x1Lims[0], x1Lims[1], nx1Res)
    yi = np.linspace(x2Lims[0], x2Lims[1], nx2Res)
    x1, x2 = np.meshgrid(xi, yi)

    x1i = np.expand_dims(x1.flatten(order='F'), axis=1)
    x2i = np.expand_dims(x2.flatten(order='F'), axis=1)
    XTest = np.concatenate((x1i, x2i), axis=1)
    preds, _, _ = predictFromCCF(CCF, XTest)

    uniquePreds, _, _  = mat_unique(preds)
    nVals        = uniquePreds.size
    numericPreds = np.empty((nVals, 1))
    numericPreds.fill(np.nan)
    numericPreds = preds
    numericPreds = np.reshape(numericPreds, (x1.shape), order='F')

    if len(n_contours_or_vals) == 0:
        if nVals >= (preds.shape[0])/2:
            # Presumably regression
            n_contours_or_vals = 50
        else:
            n_contours_or_vals = np.arange(1.5, (nVals-0.5)+1, 1)
            print(n_contours_or_vals)

    colors  = ['g', 'm', 'k', 'y']
    markers = ['x', '+', 'o', '*']

    # Plot Classes
    if Y.shape[1] != 1:
        Y = np.sum(np.multiply(Y, np.arange(1, Y.shape[1]+1)), axis=1)
    elif islogical(Y):
        Y = Y + 1

    plt.set_cmap(plt.cm.Paired)
    plt.pcolormesh(x1, x2, numericPreds)

    if plot_X:
        for k in range(1, np.max(Y)+1):
            plt.scatter(X[np.squeeze(Y==k), 0], X[np.squeeze(Y==k), 1], c=colors[k], s=4, marker=markers[k])

    plt.savefig(name, dpi=200)

    return None
