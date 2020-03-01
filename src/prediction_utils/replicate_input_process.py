import numpy as np
from utils.commonUtils import is_numeric
from utils.commonUtils import makeSureString

def replicateInputProcess(Xraw, InputProcessDetails):
    """
    This can be used to create an anonymous function that applies the same
    data transformation as was done by on the training data to new data.
    InputProcessDetails is the structure output from processInputData and
    stored in the forest.
    """
    bOrdinal = InputProcessDetails["bOrdinal"]
    Cats     = InputProcessDetails["Cats"]

    if Xraw.shape[1] != bOrdinal.size:
        assert (False), 'Incorrect number of features!'

    # TODO: Add support for dataframe
    if isinstance(XTrainRC, pd.DataFrame):
        featureNamesOrig = list(XTrainRC.columns.values)
        # Convert to Numpy
         raise NotImplementedError("To be implemented")

    X = Xraw[:, bOrdinal]

    # TODO: Maybe Impelement Expand the categorical features
    XCat = Xraw[:, ~bOrdinal]

    X = np.divide(np.subtract(X, InputProcessDetails["mu_XTrain"]), InputProcessDetails["std_XTrain"])

    if InputProcessDetails["bNaNtoMean"]:
        X[np.isnan(X)] = 0

    return X
