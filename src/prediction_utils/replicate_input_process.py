import numpy as np
import pandas as pd
from src.utils.commonUtils import is_numeric
from src.utils.commonUtils import makeSureString

def replicateInputProcess(Xraw, InputProcessDetails):
    """
    This can be used to create an anonymous function that applies the same
    data transformation as was done by on the training data to new data.
    InputProcessDetails is the structure output from processInputData and
    stored in the forest.
    """
    bOrdinal = InputProcessDetails["bOrdinal"]
    Cats     = InputProcessDetails["Cats"]
    XCat_exist = InputProcessDetails['XCat_exist']

    if Xraw.shape[1] != bOrdinal.size:
        assert (False), 'Incorrect number of features!'

    # Numerical Features
    if isinstance(Xraw, pd.DataFrame):
        # Anything not numeric in the ordinal features taken to be missing
        # values
        X = Xraw.loc[:, bOrdinal]
        bNumeric = is_numeric(X, compress=False)
        bNumeric = pd.DataFrame(bNumeric, dtype=type(True))
        X.iloc[~bNumeric] = np.nan
        X = X.to_numpy(dtype=float)
    else:
        X = Xraw[:, bOrdinal]

    # Categorical Features
    if isinstance(Xraw, pd.DataFrame) and XCat_exist:
        XCat = Xraw.loc[:, ~bOrdinal]
        XCat = makeSureString(XCat, nSigFigTol=10)
        # Expand the categorical features
        for n in range(XCat.shape[1]):
            cats_unique = Cats[n]
            nCats = len(cats_unique)
            # This is setup so that any trivial features are not included
            if nCats==1:
                continue
            sizeSoFar = X.shape[1]
            X = np.concatenate((X, np.zeros((X.shape[0], nCats))), axis=1)
            for c in range(nCats):
                X[XCat.iloc[:, n] == cats_unique[c], (sizeSoFar+c)] = 1;

    # Normalize feature vectors
    X = np.divide(np.subtract(X, InputProcessDetails["mu_XTrain"]), InputProcessDetails["std_XTrain"])

    if InputProcessDetails["bNaNtoMean"]:
        X[np.isnan(X)] = 0

    return X
