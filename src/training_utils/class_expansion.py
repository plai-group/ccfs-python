import numpy as np

def classExpansion(Y, N, optionsFor):
    """
    This function ensures that class data is in its binary expansion format

    Parameters
    ----------
    Y = Numpy array
        Class information, can be a binary expansion, a numerical
        vector of labels or a cell array of numerical or string
        labels.  For multiple inputs, should instead be a 1xV cell
        array where each cell is of a type required for single input.
    N = Float
        Number of datapoints.
    optionsFor = dict
        Forest options

    Returns
    -------
    Y = Numpy array
        Y in binary expansion format
    classes = float
        Names of classes.  In CCT only the class index is stored and
        so this is used to convert to the original name.
    optionsFor = dict
        Updated forest options, e.g. because bSepPred has been
        switched on because non-mutually exclusive classes.
    """
    # TODO: 
    if Y.shape[0] == N and Y.shape[1] == 1:
        assert (Y.shape[0] == N and Y.shape[1] == 1) ,'Seperate in-out prediction is only valid when Y is a logical array'
        [classes,~,Yindexes] = unique(Y);

        optionsFor[task_ids] = 1;

    if size(Y,1)==N && size(Y,2)==1
        assert(~optionsFor.bSepPred,'Seperate in-out prediction is only valid when Y is a logical array');
        [classes,~,Yindexes] = unique(Y);
        Y = false(size(Yindexes,1),numel(classes));
        for k=1:numel(classes)
            Y(:,k) = k==Yindexes;
        end
        optionsFor.task_ids = 1;
