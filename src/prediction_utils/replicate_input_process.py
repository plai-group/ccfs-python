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
    # if istable(Xraw):
    #     try:
    #         Xraw = table2array(Xraw)
    #     except:
    #         Xraw = table2cell(Xraw)

    X = Xraw[:, bOrdinal]

    # if isinstance(X, pd.DataFrame):
    #     # TODO: Add support for dataframe
    #     # bNumeric = is_numeric(X=X, compress=False)
    #     pass

    XCat = Xraw[:, ~bOrdinal]

    X = np.divide(np.subtract(X, InputProcessDetails["mu_XTrain"]), InputProcessDetails["std_XTrain"])

    if InputProcessDetails["bNaNtoMean"]:
        X[np.isnan(X)] = 0


    return X
