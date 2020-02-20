import logging
import numpy as np
from utils.commonUtils import is_numeric
from utils.commonUtils import fastUnique
from utils.commonUtils import sTranspose
from utils.commonUtils import queryIfColumnsVary
from utils.commonUtils import queryIfOnlyTwoUniqueRows
from utils.ccfUtils import regCCA_alt
from utils.ccfUtils import random_feature_expansion
from utils.ccfUtils import genFeatureExpansionParameters
from component_analysis import componentAnalysis
from twopoint_max_marginsplit import twoPointMaxMarginSplit

logger  = logging.getLogger(__name__)

def setupLeaf(YTrain, bReg, options):
    """
    Update tree struct to make node a leaf
    """
    tree = {}
    tree["bLeaf"]   = True
    tree["Npoints"] = YTrain.shape[0]
    tree["mean"]    = np.mean(YTrain, axis=0)

    if bReg:
        tree["std_dev"] = np.std(YTrain, axis=0, ddof=1)
        # If a mapping has been applied, invert it
        if not (options["org_stdY"].size == 0):
            tree["mean"] = tree["mean"] * options["org_stdY"]
            tree["std_dev"] = tree["std_dev"] * options["org_stdY"]

        if not (options["org_muY"].size == 0):
            tree["mean"] = tree["mean"] + options["org_muY"]

    return tree

def makeExpansionFunc(wZ, bZ, bIncOrig):
    if bIncOrig:
        f = lambda x: np.concatenate((x, random_feature_expansion(x, wZ, bZ)))
    else:
        f = lambda x: random_feature_expansion(x, wZ, bZ)

    return f

def calc_mse(cumtotal, cumsq, YTrainSort):
    value = np.divide(cumsq, (np.arange(0:YTrainSort.shape[0])).T) -\
            np.divide((cumtotal[0:-1, :]**2  + YTrainSort**2 + 2*cumtotal[0:-1, :] * YTrainSort),\
                      (np.arange(0:YTrainSort.shape[0]**2)).T)

    return value

#-------------------------------------------------------------------------------
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
    tree        = Structure containing learnt tree
    """
    # Set any missing required variables
    if (options["mseTotal"]).size == 0:
        options["mseTotal"] = YTrain.var(axis=0)

    #---------------------------------------------------------------------------
    # First do checks for whether we should immediately terminate
    #---------------------------------------------------------------------------
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

    #---------------------------------------------------------------------------
    # Subsample features as required for hyperplane sampling
    #---------------------------------------------------------------------------
    iCanBeSelected = fastUnique(X=iFeatureNum)
    iCanBeSelected[np.isnan(iCanBeSelected)] = []
    lambda_   = min(len(iCanBeSelected), options["lambda"])
    indFeatIn = np.random.randint(low=0, high=iCanBeSelected.size, size=lambda_)
    iFeatIn   = iCanBeSelected[indFeatIn]

    bInMat = np.equal(sVT(X=iFeatureNum.flatten()), np.sort(iFeatIn.flatten()))

    iIn = (np.any(bInMat, axis=0)).ravel().nonzero()[0][0]

    # Check for variation along selected dimensions and
    # resample features that have no variation
    bXVaries = queryIfColumnsVary(X=XTrain[:, iIn], tol=options["XVariationTol"])

    if not np.all(bXVaries):
        iInNew = iIn
        nSelected = 0
        iIn = iIn[bXVaries]

        while not all(bXVaries) and lambda_ > 0:
            iFeatureNum[iInNew[~bXVaries]] = np.nan
            bInMat[:, iInNew[~bXVaries]] = False
            bRemainsSelected = np.any(bInMat, aixs=1)
            nSelected = nSelected + bRemainsSelected.sum(axis=0)
            iCanBeSelected[indFeatIn] = []
            lambda_   = min(iCanBeSelected.size, options["lambda"]-nSelected)
            if lambda_ < 1:
                break
            indFeatIn = np.random.randint(low=0, high=iCanBeSelected.size, size=lambda_)
            iFeatIn   = iCanBeSelected[indFeatIn]
            bInMat    = np.equal(sVT(X=iFeatureNum.flatten()), np.sort(iFeatIn.flatten()))
            iInNew    = (np.any(bInMat, axis=0)).ravel().nonzero()[0][0]
            bXVaries  = queryIfColumnsVary(X=XTrain[:, iInNew], tol=options["XVariationTol"])
            iInNew    = np.sort(np.concatenate(iIn, iInNew[bXVaries]))

    if iIn.size == 0:
        # This means that there was no variation along any feature, therefore exit.
        tree = setupLeaf(YTrain, bReg, options)
        return tree

    #---------------------------------------------------------------------------
    # Projection bootstrap if required
    #---------------------------------------------------------------------------
    if options["bProjBoot"]:
        iTrainThis = np.random.randint(N, size=(N,1))
        XTrainBag  = XTrain[iTrainThis, iIn]
        YTrainBag  = YTrain[iTrainThis, :]
    else:
        XTrainBag = XTrain[:, iIn]
        YTrainBag = YTrain

    bXBagVaries = queryIfColumnsVary(X=XTrainBag, tol=options["XVariationTol"])

    if not np.any(bXBagVaries) or\
        (not bReg and YTrainBag.shape[1] > 1  and (np.sum(np.absolute(np.sum(YTrainBag, axis=0)) > 1e-12) < 2)) or\
        (not bReg and YTrainBag.shape[1] == 1 and np.any(np.sum(YTrainBag, axis=0) == [0, YTrainBag.shape[0]])) or\
        (bReg and np.all(var(YTrainBag) < (options["mseTotal"] * options["mseErrorTolerance"]))):
        if not options["bContinueProjBootDegenerate"]:
            tree = setupLeaf(YTrain, bReg, options)
            return tree
        else:
            XTrainBag = XTrain[:, iIn]
            YTrainBag = YTrain

    #---------------------------------------------------------------------------
    # Check for only having two points
    #---------------------------------------------------------------------------
    if (not (options["projection"].size == 0)) and ((XTrainBag.shape[0] == 1) or queryIfOnlyTwoUniqueRows(X=XTrainBag)):
        bSplit, projMat, partitionPoint = twoPointMaxMarginSplit(XTrainBag, YTrainBag, options["XVariationTol"])
        if not bSplit:
            tree = setupLeaf(YTrain, bReg, options)
            return tree

        else:
            bLessThanTrain = (XTrain[:, iIn] * projMat) <= partitionPoint
            iDir = 1
    else:
        # Generate the new features as required
        if options["bRCCA"]:
            wZ, bZ  = genFeatureExpansionParameters(XTrainBag, options["rccaNFeatures"], options["rccaLengthScale"])
            fExp    = makeExpansionFunc(wZ, bZ, options["rccaIncludeOriginal"])
            projMat, _, _ = regCCA_alt(XTrainBag, YTrainBag, options["rccaRegLambda"], options["rccaRegLambda"], 1e-8)
            if projMat.size == 0:
                projMat = np.ones((XTrainBag.shape[1], 1))
            UTrain = fExp(XTrain[:, iIn]) @ projMat

        else:
            projMat, yprojMat, _, _, _ = componentAnalysis(XTrainBag, YTrainBag, options["projections"], options["epsilonCCA"])
            UTrain = XTrain[:, iIn] @ projMat

        #-----------------------------------------------------------------------
        # Check for only having two points
        #-----------------------------------------------------------------------

        # This step catches splits based on no significant variation
        bUTrainVaries = queryIfColumnsVary(UTrain, options["XVariationTol"])

        if not np.any(bUTrainVaries):
            tree = setupLeaf(YTrain,bReg,options);

        UTrain  = UTrain[:, bUTrainVaries]
        projMat = projMat[:, bUTrainVaries]

        if options["bUseOutputComponentsMSE"] and bReg and (YTrain.shape[1] > 1) and\
           (not (yprojMat.size == 0)) and (options["splitCriterion"] == 'mse'):
           VTrain = YTrain @ yprojMat

        
