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
    Cats = InputProcessDetails["Cats"]

    if size(Xraw.shape[1] != bOrdinal.size:
        assert (True), 'Incorrect number of features'

    # #  Add support for dataframe
    # if istable(Xraw):
    #     try:
    #         Xraw = table2array(Xraw)
    #     except:
    #         Xraw = table2cell(Xraw)

    X = Xraw[:, bOrdinal]
    if type(f) is dict:
        bNumeric = is_numeric(X)

    XCat = Xraw[:, ~bOrdinal]
    if iscell(XCat)
        XCat = makeSureString(XCat,10);
    end

    for n in range(XCat.shape[1]):
        nCats = Cats[n].size

        if nCats == 1:
            continue

        sizeSoFar = X.shape[1]

        X = [X, np.zeros((X.shape[0], nCats))]
        for c in range(nCats):
            if iscell(Cats{n})
                X(strcmp(XCat(:,n),Cats{n}{c}),(sizeSoFar+c)) = 1;
            else
                X(XCat(:,n)==Cats{n}(c),(sizeSoFar+c)) = 1;
            end
        end
    end

    X = np.divide(np.subtract(X, InputProcessDetails["mu_XTrain"]), InputProcessDetails["std_XTrain"])

    if InputProcessDetails["bNaNtoMean"]:
        X[np.isnan(X)] = 0


    return X
