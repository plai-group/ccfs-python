import numpy as np
import pandas as pd
from src.utils.commonUtils import sVT
from src.utils.commonUtils import is_numeric
from src.utils.commonUtils import makeSureString
from src.prediction_utils.replicate_input_process import replicateInputProcess

def processInputData(XTrainRC, bOrdinal=None, XTestRC=None, bNaNtoMean=False):
    """
    Process input features, expanding categoricals and converting to zScores.

    Parameters
    ----------
    XTrain: Unprocessed input features, can be a numerical array or dataframe.
            Each row is a seperate data point.
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
        featureNamesOrig = np.array(list(XTrainRC.columns.values))
        # Rename pandas column for indexing convenience
        new_col_names = [idx for idx in range(len(featureNamesOrig))]
        XTrainRC.columns = new_col_names
    else:
        featureNamesOrig = np.array([f'Var_{idx}' for idx in range(XTrainRC.shape[1])])

    # Logical array indicating if the corresponding feature is ordinal
    if bOrdinal == None:
        if isinstance(XTrainRC, type(np.array([]))):
            # Default is that if input is all numeric, everything is treated as
            # ordinal
            bOrdinal = np.array([True] * D)
        else:
            # Numeric features treated as ordinal
            bNumeric = is_numeric(XTrainRC, compress=False)
            # Features with more than one unique string taken as non-ordinal
            iContainsString = (np.sum(~bNumeric, axis=0) > 0).ravel().nonzero()[0]
            nStr = np.zeros((XTrainRC.shape[1]), dtype=int)
            for n in iContainsString.flatten(order='F'):
                x_unique = np.unique(XTrainRC.loc[~bNumeric[:, n], n])
                nStr[n]  = len(x_unique)
            bOrdinal = nStr < 2
            # Features with only a single unqiue string and otherwise
            # numeric treated also treated as ordinal with the string
            # taken to give a missing value
            iSingleStr = (nStr == 1).ravel().nonzero()[0]
            for n in iSingleStr:
                XTrainRC.loc[~bNumeric[:, n], n] = np.nan

    elif len(bOrdinal) != XTrainRC.shape[1]:
        assert (True), 'bOrdinal must match size of XTrainRC!'

    # Numerical Features
    if isinstance(XTrainRC, pd.DataFrame):
        # Anything not numeric in the ordinal features taken to be missing
        # values
        XTrain   = XTrainRC.loc[:, bOrdinal]
        bNumeric = is_numeric(XTrain, compress=False)
        bNumeric = pd.DataFrame(bNumeric, dtype=type(True))
        XTrain[~bNumeric] = np.nan
        XTrain = XTrain.to_numpy(dtype=float)
    else:
        XTrain = XTrainRC[:, bOrdinal]

    # Categorical Features
    if isinstance(XTrainRC, pd.DataFrame):
        XCat = XTrainRC.loc[:, ~bOrdinal]
        XCat = makeSureString(XCat, nSigFigTol=10)
        # Previous properties
        iFeatureNum  = list(range(XTrain.shape[1]))
        featureNames = featureNamesOrig[bOrdinal]
        featureBaseNames = featureNamesOrig[~bOrdinal]
        # Collect Categorical features
        Cats = {}
        iFeatureNum = np.array([iFeatureNum], dtype=int)
        # Expand the categorical features
        for n in range(XCat.shape[1]):
            cats_unique = np.unique(XCat.iloc[:, n])
            Cats[n]  = cats_unique
            newNames = np.array([f'Cat_{name}' for name in cats_unique])
            featureNames = np.concatenate((featureNames, newNames))

            nCats = len(cats_unique)
            # This is setup so that any trivial features are not included
            if nCats==1:
                continue
            sizeSoFar = iFeatureNum.shape[1]
            if len(iFeatureNum) == 0:
                iFeatureNum = np.ones((1,nCats))
            else:
                iFeatureNum = np.concatenate((iFeatureNum, (iFeatureNum[:, -1] + 1) * np.ones((1,nCats))), axis=1).astype(float)

            XTrain = np.concatenate((XTrain, np.zeros((XTrain.shape[0], nCats))), axis=1)
            for c in range(nCats):
                XTrain[XCat.iloc[:, n] == cats_unique[c], (sizeSoFar+c)] = 1;

        # Remove single dimension
        iFeatureNum = np.squeeze(iFeatureNum)
    else:
        Cats = {}
        iFeatureNum  = np.arange(XTrain.shape[1]) * 1.0
        featureNames = featureNamesOrig[bOrdinal]
        featureBaseNames = featureNamesOrig[~bOrdinal]

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
    inputProcessDetails["Cats"]       = Cats # {0: array(['False', 'True'], dtype=object), 1: array(['f', 't'], dtype=object)}
    inputProcessDetails['bOrdinal']   = bOrdinal
    inputProcessDetails['mu_XTrain']  = mu_XTrain
    inputProcessDetails['std_XTrain'] = std_XTrain
    inputProcessDetails['bNaNtoMean'] = bNaNtoMean

    if XTestRC == None:
        return XTrain, iFeatureNum, inputProcessDetails, featureNames

    XTest = replicateInputProcess(Xraw=XTestRC, InputProcessDetails=inputProcessDetails)

    return XTrain, iFeatureNum, inputProcessDetails, XTest, featureNames
