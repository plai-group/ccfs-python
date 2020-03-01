import numpy as np
from predict_from_cct import predictFromCCT
from prediction_utils.replicate_input_process import replicateInputProcess
from prediction_utils.tree_output_forest_pred import treeOutputsToForestPredicts

def predictFromCCF(CCF, X):
    """
    Parameters
    ----------
    CF: Output from genCCF.  This is a structure with a field
        Trees, giving a cell array of tree structures, and
        options which is an object of type optionsClassCCF
    X:  Input features at which to make predictions, each row should be a seperate data point

    Returns
    -------
    forestPredicts: Array of numeric predictions corresponding to the prediction.
                    - For regression this is a vector of scalar predictions
                    - For multivariate regression it is a matrix were each
                      column is a different output variable.
                    - For classification where Y was original provided as
                      a numeric array then it is a vector predicted class labels
                    - For classification where Y was originally a logical
                      array or cell array of strings then is a vector of
                      indices to CCF.classNames
                    - For multiple output classification then it is a
                      matrix where columns correspond to the output of the
                      appropriate single output classification case.
    forestProbs: Assigned probability to each class for classification.
                 Empty for regression.  Columns are different indexes to CCF.classNames
    treeOutputs: Individual tree outputs stored as a NxLxK array where N is
                 the number of predicted data points, L is the number of
                 trees and K is the number of predictions.  K=1 for
                 regression, K = number of classes for classification, and
                 for regression then each output is concatenated in the third dimension.
    """
    X = replicateInputProcess(X, CCF["inputProcessDetails"])

    nTrees = len(CCF["Trees"])

    # Preallocate output space
    pcctx0 = predictFromCCT(CCF["Trees"][0], X)[0]
    pcctx0 = np.expand_dims(pcctx0, axis=1)
    treeOutputs = np.tile(pcctx0, [1, nTrees, 1])

    for n in range(1, nTrees):
        treeOutputs[:, n, :], _ = predictFromCCT(CCF["Trees"][n], X)

    forestPredicts, forestProbs = treeOutputsToForestPredicts(CCF, treeOutputs)

    return forestPredicts, forestProbs, treeOutputs
