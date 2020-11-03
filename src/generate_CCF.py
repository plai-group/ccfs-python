import numpy as np
import multiprocessing as mp
from collections import OrderedDict
from src.utils.commonUtils import fastUnique
from src.utils.commonUtils import is_numeric
from src.utils.ccfUtils import pcaLite
from src.utils.ccfUtils import randomRotation
from src.utils.ccfUtils import random_missing_vals
from src.predict_from_CCT import predictFromCCT
from src.training_utils.grow_CCT import growCCT
from src.training_utils.class_expansion import classExpansion
from src.training_utils.process_inputData import processInputData
from src.training_utils.rotation_forest_DP import rotationForestDataProcess
from src.prediction_utils.replicate_input_process import replicateInputProcess

import logging
logger  = logging.getLogger(__name__)

#-------------------------------------------------------------------------------#
def updateForD(optionsFor, D):
    """
    Updates the options for a particular D to set lambda and decide whether to
    do projection bootstrapping (by default this is done unless lambda==D).
    """
    if optionsFor["lambda"] == 'sqrt':
        optionsFor["lambda"] = np.ceil(np.sqrt(D))
    elif optionsFor["lambda"] == 'log':
        if int(D) == 3:
            optionsFor["lambda"] = 2
        else:
            optionsFor["lambda"] = np.ceil(np.log2(D) + 1)
    elif optionsFor["lambda"] == 'all':
        optionsFor["lambda"] = D
    elif (not is_numeric(optionsFor["lambda"])):
        logger.warning('Invalid option set for lambda')

    if optionsFor["bProjBoot"] == 'default':
        if int(D) <= optionsFor["lambda"]:
            optionsFor["bProjBoot"] = False
        else:
            optionsFor["bProjBoot"] = True

    if optionsFor["bBagTrees"] == 'default':
        if int(D) <= optionsFor["lambda"]:
            optionsFor["bBagTrees"] = True
        else:
            optionsFor["bBagTrees"] = False

    return optionsFor


#-------------------------------------------------------------------------------#
def genTree(XTrain, YTrain, bReg, optionsFor, iFeatureNum, Ntrain):
    """
    A sub-function is used so that it can be shared between the for and
    parfor loops.  Does required preprocessing such as randomly setting
    missing values, then calls the tree training function
    """
    if optionsFor["missingValuesMethod"] == 'random':
        # Randomly set the missing values.  This will be different for each tree
        XTrain = random_missing_vals(XTrain)

    N = XTrain.shape[0]

    # Bag if required
    if optionsFor["bBagTrees"] or (Ntrain != N):
        all_samples = np.arange(N)
        iTrainThis  = np.random.choice(all_samples, Ntrain, replace=optionsFor["bBagTrees"])
        iOob        = np.setdiff1d(all_samples, iTrainThis).T
        XTrainOrig  = XTrain
        XTrain      = XTrain[iTrainThis, :]
        YTrain      = YTrain[iTrainThis, :]

    # Apply pre rotations if any requested.  Note that these all include a
    # subtracting a the mean prior to the projection (because this is a natural
    # part of pca) and this is therefore replicated at test time
    if optionsFor["treeRotation"] == 'rotationForest':
        # This allows functionality to use the Rotation Forest algorithm as a
        # meta method for individual CCTs
        prop_classes_eliminate = optionsFor["RotForpClassLeaveOut"]
        if bReg:
            prop_classes_eliminate = 0
        R, muX, XTrain = rotationForestDataProcess(XTrain, YTrain, optionsFor["RotForM"], optionsFor["RotForpS"], prop_classes_eliminate)

    elif optionsFor["treeRotation"] == 'random':
        muX = np.nanmean(XTrain, axis=0)
        R   = randomRotation(N=XTrain.shape[1])
        XTrain = np.dot(np.subtract(XTrain, muX), R)

    elif optionsFor["treeRotation"] == 'pca':
        R, muX, XTrain = pcaLite(XTrain, False, False)

    # Train the tree
    tree = growCCT(XTrain, YTrain, bReg, optionsFor, iFeatureNum, 0)
    
    # Calculate out of bag error if relevant
    if optionsFor["bBagTrees"]:
        tree["iOutOfBag"] = iOob
        tree["predictsOutOfBag"], _ = predictFromCCT(tree, XTrainOrig[iOob, :])

    # Store rotation deatils if necessary
    if not (optionsFor["treeRotation"] == None):
        tree["rotDetails"] = {'R': R, 'muX': muX}

    return tree


#-------------------------------------------------------------------------------#
def genTree_parallel(XTrain, YTrain, bReg, optionsFor, iFeatureNum, Ntrain, pos):
    """
    A sub-function is used so that it can be shared between the for and
    parfor loops.  Does required preprocessing such as randomly setting
    missing values, then calls the tree training function
    """
    if optionsFor["missingValuesMethod"] == 'random':
        # Randomly set the missing values.  This will be different for each tree
        XTrain = random_missing_vals(XTrain)

    N = XTrain.shape[0]

    # Bag if required
    if optionsFor["bBagTrees"] or (Ntrain != N):
        all_samples = np.arange(N)
        iTrainThis  = np.random.choice(all_samples, Ntrain, replace=optionsFor["bBagTrees"])
        iOob        = np.setdiff1d(all_samples, iTrainThis).T
        XTrainOrig  = XTrain
        XTrain      = XTrain[iTrainThis, :]
        YTrain      = YTrain[iTrainThis, :]

    # Apply pre rotations if any requested.  Note that these all include a
    # subtracting a the mean prior to the projection (because this is a natural
    # part of pca) and this is therefore replicated at test time
    if optionsFor["treeRotation"] == 'rotationForest':
        # This allows functionality to use the Rotation Forest algorithm as a
        # meta method for individual CCTs
        prop_classes_eliminate = optionsFor["RotForpClassLeaveOut"]
        if bReg:
            prop_classes_eliminate = 0
        R, muX, XTrain = rotationForestDataProcess(XTrain, YTrain, optionsFor["RotForM"], optionsFor["RotForpS"], prop_classes_eliminate)

    elif optionsFor["treeRotation"] == 'random':
        muX = np.nanmean(XTrain, axis=0)
        R   = randomRotation(N=XTrain.shape[1])
        XTrain = np.dot(np.subtract(XTrain, muX), R)

    elif optionsFor["treeRotation"] == 'pca':
        R, muX, XTrain = pcaLite(XTrain, False, False)

    # Train the tree
    tree = growCCT(XTrain, YTrain, bReg, optionsFor, iFeatureNum, 0)

    # Calculate out of bag error if relevant
    if optionsFor["bBagTrees"]:
        tree["iOutOfBag"] = iOob
        tree["predictsOutOfBag"], _ = predictFromCCT(tree, XTrainOrig[iOob, :])

    # Store rotation deatils if necessary
    if not (optionsFor["treeRotation"] == None):
        tree["rotDetails"] = {'R': R, 'muX': muX}

    return (pos, tree)


#-------------------------------------------------------------------------------#
def genCCF(XTrain, YTrain, nTrees=500, bReg=False, optionsFor={}, do_parallel=False, XTest=None, bKeepTrees=True, iFeatureNum=None, bOrdinal=None):
    """
    Creates a canonical correlation forest (CCF) comprising of nTrees
    canonical correlation trees (CCT) containing splits based on the a CCA
    analysis between the training data and a binary representation of the
    class labels.

    Parameters
    ----------
    nTrees: Int
            Number of trees to create.
    XTrain: pandas DataFrame/Numpy array
            Numpy array => For Numeric only
            Pandas DataFrame => For Numeric/String/Categorical
            Array giving training features. Each row should be a
            seperate data point and each column a seperate feature.
            Must be numerical array with missing values marked as
            NaN if iFeatureNum is provided, otherwise can be any
            format accepted by processInputData function

   YTrain: Numpy array
           Output data.
           Regression:
             Column vector of outputs
           Multivariate regression:
             Matrix where each column is a different output
           Classification:
             There three excepted formats: a column vector of
             integers representing class labels, a cell column
             vector of strings giving the class name, or a NxK
             logical array representing a 1-of-K encoding of the
             class labels.  For binary classification a logical
             column vector is also accepted.
           Multi-output classification:
              Two accepted formats: a NxNout array of numeric
              class labels where each column is a different
              output or a 1xNout cell array where each cell is a
              seperate input satisfying the requirements for a
              single classification.
   bReg: Boolean
         Whether to perform regression instead of classification.
         Default = False (i.e. classification).


    Advanced usage:
    ---------------
    options: dict
             Options dict created by optionsClassCCF.  If left
             blank then a default set of options corresponding to the
             method detailed in the paper is used.
     XTest: Numpy array
             Test data to make predictions for.  If the input
             features for the test data are known at test time then
             using this input with the option bKeepTrees = false can
             significantly reduce the memory requirement.
     bKeepTrees: Boolean
             If false and XTest is given then the individual trees
             are not stored in order to save memory.  Default = true
     iFeatureNum:
             Vector for grouping of categorical variables as
             generated by processInputData function.  If left blank
             then the data is processed using processInputData.
     bOrdinal:
             If the data is to be processed, this allows
             specification of ordinal variables.  For default
             behaviour see process_inputData.py

    Returns
    -------
     CCF:  dict
           Structure with following fields
           - Trees = Cell array of CCTs
           - bReg = Whether a regression CCF
           - options = The used options structure (after
                       processing for things like task_ids based on the data)
           - inputProcessDetails = Details required to replicate
                       the input feature transforms (e.g. converting to
                       z-scores) done during training
           - outOfBagError = If bagging was used, gives the
                       average out of bag error.  Otherwise empty.
           - timing_stats = see bCalcTimingStats option in optionsClassCCF.m
     forPred: Numpy Array
           Forest predictions for XTest if provided
     forProbs: Numpy Array
           Forest probabilities for XTest if provided
     treeOutputs: Numpy Array
           Individual tree predictiosn for XTest if provided, see predictFromCCF
    """
    bNaNtoMean = (optionsFor['missingValuesMethod'] == 'mean')

    if iFeatureNum == None:
        iFeatureNum=np.array([]) # Create empty array

    # Process input data
    if (not is_numeric(XTrain)) or (not (iFeatureNum == None)) or (iFeatureNum.size == 0):
        # If XTrain not in numeric form or if a grouping of features is not
        # provided, apply the input data processing.
        if (not (iFeatureNum.size == 0)):
            logger.warning('iFeatureNum provided but XTrain not in array format, over-riding')
        if (XTest == None):
            XTrain, iFeatureNum, inputProcessDetails, _  = processInputData(XTrainRC=XTrain, bOrdinal=bOrdinal, XTestRC=None, bNaNtoMean=bNaNtoMean)
        else:
            XTrain, iFeatureNum, inputProcessDetails, XTest, _ = processInputData(XTrainRC=XTrain, bOrdinal=bOrdinal, XTest=XTest, bNaNtoMean=bNaNtoMean)

    else:
        # Process inputs, e.g. converting categoricals and converting to z-scores
        mu_XTrain  = np.nanmean(XTrain, axis=0)
        std_XTrain = np.nanstd(XTrain,  axis=0, ddof=1)
        inputProcessDetails = {'bOrdinal': np.array([True] * XTrain.shape[1]), 'mu_XTrain': mu_XTrain, 'std_XTrain': std_XTrain}
        inputProcessDetails["Cats"] = {}
        inputProcessDetails['XCat_exist'] = False
        XTrain = replicateInputProcess(XTrain, inputProcessDetails)
        if (not (XTest.size == 0)):
             XTest = replicateInputProcess(XTest, inputProcessDetails)
    

    N = XTrain.shape[0]
    # Note that setting of number of features to subsample is based only
    # number of features before expansion of categoricals.
    D = (fastUnique(iFeatureNum)).size

    if (not bReg):
        # Process provided classes
        YTrain, classes, optionsFor = classExpansion(Y=YTrain, N=N, optionsFor=optionsFor)

        if classes.size == 1:
            logger.warning('Only 1 class present in training data!');

        optionsFor = updateForD(optionsFor, D)

        # Remove any single dims
        #YTrain = np.squeeze(YTrain)
        
        # Stored class names can be used to link the ids given in the CCT to the
        # actual class names
        optionsFor["classNames"] = classes

    else:
        # Center and normalize the outputs for regression for numerical
        # reasons, this is undone in the predictors
        muY  = np.mean(YTrain)
        stdY = np.std(YTrain, axis=0, ddof=1)

        # For now just set stdY to be 1 instead of zero to prevent NaNs if a
        # dimensions has no variation.
        stdY[stdY==0] = 1

        YTrain = np.divide(np.subtract(YTrain, muY), stdY)

        optionsFor = updateForD(optionsFor, D)
        optionsFor["org_muY"]  = muY
        optionsFor["org_stdY"] = stdY
        optionsFor["mseTotal"] = 1
    
    # Fill in any unset projection fields and set to false
    projection_fields = ['CCA', 'PCA', 'CCAclasswise', 'Original', 'Random']
    all_fields = optionsFor["projections"].keys()
    for npf in projection_fields:
        if npf not in all_fields:
            optionsFor["projections"][npf] = False

    if not bKeepTrees:
        bKeepTrees = True
        logger.warning('Selected not to keep trees but only requested a single output of the trees, reseting bKeepTrees to true')

    if XTest == None:
        XTest = np.empty((0, XTrain.shape[0]))
        XTest.fill(np.nan)

    forest = OrderedDict()

    treeOutputTest = np.empty((XTest.shape[0], nTrees, YTrain.shape[1]))
    treeOutputTest.fill(np.nan)
    n_nodes_trees = np.empty((nTrees, 1))
    n_nodes_trees.fill(np.nan)

    Ntrain = int(N * optionsFor["propTrain"])

    # Train the trees
    if do_parallel:
        pool = mp.Pool(processes=mp.cpu_count())
        processes = [pool.apply_async(genTree_parallel, args=(XTrain, YTrain, bReg, optionsFor, iFeatureNum, Ntrain, n_i)) for n_i in range(nTrees)]

        # Get process results
        all_trees = [p.get() for p in processes]
        all_trees.sort() # Sort the results by pos

        # Collect
        for nT in range(nTrees):
            if bKeepTrees:
                forest[nT] = all_trees[nT][1]

    else:
        for nT in range(nTrees):
            # Generate tree
            tree = genTree(XTrain, YTrain, bReg, optionsFor, iFeatureNum, Ntrain)

            if bKeepTrees:
                forest[nT] = tree

            if nT%25 == 0:
                print('Progress: {}/{}'.format(nT, nTrees))
            
            del tree

    print('Completed')
    print('..................................................................')

    # Setup outputs
    CCF = {}
    CCF["Trees"]   = forest
    CCF["bReg"]    = bReg
    CCF["options"] = optionsFor
    CCF["inputProcessDetails"] = inputProcessDetails
    CCF["classNames"] = optionsFor["classNames"]

    if optionsFor["bBagTrees"] and bKeepTrees:
        # Calculate the out of back error if relevant
        cumOOb = np.zeros((YTrain.shape[0], (CCF["Trees"][0]["predictsOutOfBag"]).shape[1]))
        nOOb   = np.zeros((YTrain.shape[0], 1))
        for nTO in range(len(CCF["Trees"])):
            cumOOb[CCF["Trees"][nTO]["iOutOfBag"], :] = cumOOb[CCF["Trees"][nTO]["iOutOfBag"], :] + CCF["Trees"][nTO]["predictsOutOfBag"]
            nOOb[CCF["Trees"][nTO]["iOutOfBag"]] = nOOb[CCF["Trees"][nTO]["iOutOfBag"]] + 1
        oobPreds = np.divide(cumOOb, nOOb)
        if bReg:
            CCF["outOfBagError"] = np.nanmean((oobPreds - np.add(np.multiply(YTrain, stdY), muY))**2, axis=0)
        elif optionsFor["bSepPred"]:
            CCF["outOfBagError"] = (1 - np.nanmean((oobPreds > 0.5) == YTrain, axis=0))
        else:
            # Check if task_ids is single number
            if type(optionsFor["task_ids"]) == int:
                task_ids_size = 1
                forPreds = np.empty((XTrain.shape[0], 1))
                forPreds.fill(np.nan)
                YTrainCollapsed = np.empty((XTrain.shape[0], 1))
                YTrainCollapsed.fill(np.nan)
                forPreds[:, -1]        = np.argmax(oobPreds[:, optionsFor["task_ids"][-1]:], axis=1)
                YTrainCollapsed[:, -1] = np.argmax(  YTrain[:, optionsFor["task_ids"][-1]:], axis=1)
                CCF["outOfBagError"]   = (1 - np.nanmean(forPreds==YTrainCollapsed, axis=0))
            else:
                forPreds = np.empty((XTrain.shape[0], optionsFor["task_ids"].size))
                forPreds.fill(np.nan)
                YTrainCollapsed = np.empty((XTrain.shape[0], optionsFor["task_ids"].size))
                YTrainCollapsed.fill(np.nan)
                for nO in range(optionsFor["task_ids"].size - 2):
                    forPreds[:, nO]        = np.argmax(oobPreds[:, optionsFor["task_ids"][nO]:optionsFor["task_ids"][nO+1]-1], axis=1)
                    YTrainCollapsed[:, nO] = np.argmax(  YTrain[:, optionsFor["task_ids"][nO]:optionsFor["task_ids"][nO+1]-1], axis=1)
                forPreds[:, -1]        = np.argmax(oobPreds[:, optionsFor["task_ids"][-1]:], axis=1)
                YTrainCollapsed[:, -1] = np.argmax(  YTrain[:, optionsFor["task_ids"][-1]:], axis=1)
                CCF["outOfBagError"]   = (1 - np.nanmean(forPreds==YTrainCollapsed, axis=0))
    else:
        CCF["outOfBagError"] = 'OOB error only returned if bagging used and trees kept.\
                                Please use CCF-Bag instead via options=optionsClassCCF.defaultOptionsCCFBag!'

    return CCF #, forestPredictsTest, forestProbsTest, treeOutputTest
