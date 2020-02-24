import numpy as np
import pandas as pd
#from prediction_utils.replicate_input_process import replicateInputProcess

def processInputData(XTrainRC, bOrdinal=None, XTestRC=None, bNaNtoMean=False):
    """
    XTrainRC: Numpy array or dataframe
    """
    D = XTrainRC.shape[1]

    if isinstance(XTrainRC, pd.DataFrame):
        featureNamesOrig = list(XTrainRC.columns.values)
        # Convert to Numpy
        XTrainRC = XTrainRC.to_numpy()
    else:
        featureNamesOrig = np.array(['Var'] * XTrainRC.shape[1])

    if bOrdinal == None:
        if isinstance(XTrainRC, type(np.array([]))):
            bOrdinal = np.array([True] * D)
        else:
            # TODO: Add Support for DataFrame as well
            assert (True), 'Current support for Numpy array only!'
    elif len(bOrdinal) != XTrainRC.shape[1]:
        assert (True), 'bOrdinal must match size of XTrainRC!'

    XTrain = XTrainRC[:, bOrdinal]

    iFeatureNum  = np.arange(XTrain.shape[1])
    featureNames = featureNamesOrig[bOrdinal]
    featureBaseNames = featureNamesOrig[~bOrdinal]

    # Convert to Z-scores, Normalize feature vectors
    mu_XTrain  = np.nanmean(XTrain, axis=0)
    std_XTrain = np.nanstd(XTrain, axis=0, ddof=1)
    std_XTrain[abs(std_XTrain)<1e-10] = 1.0
    XTrain = np.divide(np.subtract(XTrain, mu_XTrain), std_XTrain)

    if bNaNtoMean:
        XTrain[np.isnan(XTrain)] = 0

    # TODO: Expand the categorical features
    Cats = np.array([])
    XCat = XTrainRC[:, ~bOrdinal]

    # if iscell(XCat)
    #     XCat = makeSureString(XCat,10);

    # If required, generate function for converting additional data and
    # calculate conversion for any test data provided.
    inputProcessDetails = {}
    inputProcessDetails["Cats"]       = Cats
    inputProcessDetails['bOrdinal']   = bOrdinal
    inputProcessDetails['mu_XTrain']  = mu_XTrain
    inputProcessDetails['std_XTrain'] = std_XTrain
    inputProcessDetails['bNaNtoMean'] = bNaNtoMean

    if XTestRC == None:
        return XTrain, iFeatureNum, inputProcessDetails, featureNames

    XTest = replicateInputProcess(Xraw=XTestRC, InputProcessDetails=inputProcessDetails);

    return XTrain, iFeatureNum, inputProcessDetails, XTest, featureNames
