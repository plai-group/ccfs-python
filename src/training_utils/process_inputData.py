import numpy as np
import pandas as pd
from src.utils.commonUtils import sVT
from src.utils.commonUtils import is_numeric
from src.prediction_utils.replicate_input_process import replicateInputProcess

def processInputData(XTrainRC, bOrdinal=None, XTestRC=None, bNaNtoMean=False):
    """
    Process input features, expanding categoricals and converting to zScores.

    Parameters
    ----------
    XTrain: Unprocessed input features, can be a numerical array, a cell
            array or a table.  Each row is a seperate data point.
    bOrdinal: Logical array indicating if the corresponding feature is
              ordinal (i.e. numerical or order categorical).  The default
              treatment is that numerical inputs are ordinal and strings
              are not.  If a feature contains for numerical features and
              strings, it is presumed to not be ordinal unless there is
              only a single string category, in which case this is
              presumed to indicate missing data.
    XTest: Additional data to be transformed.  This is seperate to the training
           data for the purpose of Z-scores and to avoid using any features /
           categories that do not appear in the training data.
    bNaNtoMean: Replace NaNs with the mean, default false;

    Returns
    -------
    XTrain: Processed input features
    iFeatureNum: Array idenitifying groups of expanded features.
                 Expanded features with the same iFeatureNum value come
                 from the same original non-ordinal feature.
    inputProcessDetails: Details required to convert new data in the same way
    XTest: Additional processed input features
    featureNames: Names of the expanded features.  Variable names are
                  taken from the table properties if in the input is a
                  cell.  For expanded categorical features the name of the
                  category is also included.
    """
    D = XTrainRC.shape[1]


    if isinstance(XTrainRC, pd.DataFrame):
        featureNamesOrig = list(XTrainRC.columns.values)
        # Convert to Numpy
        raise NotImplementedError("To be implemented")
    else:
        featureNamesOrig = np.array(['Var'] * XTrainRC.shape[1])


    if bOrdinal == None:
        if isinstance(XTrainRC, type(np.array([]))):
            # Default is that if input is all numeric, everything is treated as
            # ordinal
            bOrdinal = np.array([True] * D)

        else:
            # Numeric features treated as ordinal, features with only a single
            # unqiue string and otherwise numeric treated also treated as
            # ordinal with the string taken to give a missing value and
            # features with more than one unique string taken as non-ordinal
            bNumeric = is_numeric(X, compress=False)
            iContainsString = (np.sum(~bNumeric, axis=0) > 0).ravel().nonzero()[0]
            nStr = np.zeros((1, XTrainRC.shape[1]))
            for n in iContainsString.flatten(order='F'):
                xtrain_unique = np.unique(XTrainRC[~bNumeric[:, n], n])
                nStr[:, n] = xtrain_unique.size

            bOrdinal   = nStr < 2
            iSingleStr = (nStr == 1).ravel().nonzero()[0]
            for n in range(iSingleStr.size):
                XTrainRC[~bNumeric[:, iSingleStr[n]], iSingleStr[n]] = np.array([np.nan])

    elif len(bOrdinal) != XTrainRC.shape[1]:
        assert (True), 'bOrdinal must match size of XTrainRC!'

    XTrain = XTrainRC[:, bOrdinal]
    XCat   = XTrainRC[:, ~bOrdinal]

    iFeatureNum  = np.arange(XTrain.shape[1]) * 1.0
    featureNames = featureNamesOrig[bOrdinal]
    featureBaseNames = featureNamesOrig[~bOrdinal]

    # TODO: Maybe Impelement Expand the categorical features
    Cats = np.array([])

    # Convert to Z-scores, Normalize feature vectors
    mu_XTrain  = np.nanmean(XTrain, axis=0)
    std_XTrain = np.nanstd(XTrain, axis=0, ddof=1)
    std_XTrain[abs(std_XTrain)<1e-10] = 1.0
    XTrain = np.divide(np.subtract(XTrain, mu_XTrain), std_XTrain)

    if bNaNtoMean:
        XTrain[np.isnan(XTrain)] = 0

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

    XTest = replicateInputProcess(Xraw=XTestRC, InputProcessDetails=inputProcessDetails)

    return XTrain, iFeatureNum, inputProcessDetails, XTest, featureNames
