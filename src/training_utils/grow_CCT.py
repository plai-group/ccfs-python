import logging
import numpy as np
from utils.commonUtils import is_numeric
logger  = logging.getLogger(__name__)

def growCCT(XTrain, YTrain, bReg, options, iFeatureNum, depth):
    """
    This function applies greedy splitting according to the CCT algorithm and the
    provided options structure. Algorithm either returns a leaf or forms an
    internal splitting node in which case the function recursively calls itself
    for each of the children, eventually returning the corresponding subtree.

    Parameters
    ----------
    XTrain      = Array giving training features.  Data should be
                  processed using processInputData before being passed to
                  CCT
    YTrain      = Output data after formatting carried out by genCCF
    bReg        = Whether to perform regression instead of classification.
                  Default = false (i.e. classification).
    options     = Options class of type optionsClassCCF.  Some fields are
                  updated during recursion
    iFeatureNum = Grouping of features as per processInputData.  During
                  recursion if a feature is found to be identical across
                  data points, the corresponding values in iFeatureNum are
                  replaced with NaNs.
    depth       = Current tree depth (zero based)

    Returns
    -------
    tree         = Structure containing learnt tree
    """
    # Set any missing required variables
    if (options["mseTotal"]).size == 0:
        options["mseTotal"] = YTrain.var(axis=0)

    """
    First do checks for whether we should immediately terminate
    """
    N = XTrain.shape[0]
    # Return if one training point, pure node or if options for returning
    # fulfilled.  A little case to deal with a binary YTrain is required.
    bStop = (N < (np.amax([2, options["minPointsForSplit"], 2 * options["minPointsLeaf"]]))) or\
            (is_numeric(options["maxDepthSplit"]) and depth > options["maxDepthSplit"])

    if depth > 490 and strcmpi(options["maxDepthSplit"], 'stack'):
        bStop = True
        logging.warning('Reached maximum depth imposed by stack limitations!')

    if bStop:
        tree = setupLeaf(YTrain, bReg, options)

        return tree
    elif:
        # Check class variation
        sumY = np.sum(YTrain, axis=0)
        bYVaries = (sumY ~= 0) and (sumY ~= N)
        if ~(np.any(bYVaries)):
            tree = setupLeaf(YTrain,bReg,options);
            return tree
    else:
        # Check if variance in Y is less than the cut off amount
         varY = YTrain.var(axis=0)
         if np.all(varY < (options["mseTotal"] * options["mseErrorTolerance"])):
             tree = setupLeaf(YTrain, bReg, options)
             return tree

    """
    Subsample features as required for hyperplane sampling
    """
    
