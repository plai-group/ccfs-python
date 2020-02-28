import scipy.io
import numpy as np
from collections  import OrderedDict
from generate_CCF import genCCF
from predict_from_ccf import predictFromCCF
from plotting.plot_surface import plotCCFDecisionSurface

# Sample Spital Data Testing script

# ----------------------Use optionsClassCCF---------------------------------#
# Default HyperParams
optionsClassCCF = {}
optionsClassCCF['lambda'] = 2
optionsClassCCF['splitCriterion'] = 'info'
optionsClassCCF['minPointsLeaf'] = 1
optionsClassCCF['bUseParallel'] = 1
optionsClassCCF['bCalcTimingStats'] = 1
optionsClassCCF['bSepPred'] = False
optionsClassCCF['taskWeights'] = 'even'
optionsClassCCF['bProjBoot'] = False
optionsClassCCF['bBagTrees'] = True
optionsClassCCF['projections'] = OrderedDict() # To ensure consistent order
optionsClassCCF['projections']['CCA'] = True
optionsClassCCF['treeRotation'] = None
optionsClassCCF['propTrain'] = 1
optionsClassCCF['epsilonCCA'] = 1.0000e-04
optionsClassCCF['mseErrorTolerance'] = 1.0000e-06
optionsClassCCF['maxDepthSplit'] = 'stack'
optionsClassCCF['XVariationTol'] = 1.0e-10
optionsClassCCF['RotForM'] = 3
optionsClassCCF['RotForpS'] = 0.7500
optionsClassCCF['RotForpClassLeaveOut'] = 0.5000
optionsClassCCF['minPointsForSplit'] = 2
optionsClassCCF['dirIfEqual'] = 'first'
optionsClassCCF['bContinueProjBootDegenerate'] = 1
optionsClassCCF['multiTaskGainCombination'] = 'mean'
optionsClassCCF['missingValuesMethod'] = 'random'
optionsClassCCF['bUseOutputComponentsMSE'] = 0
optionsClassCCF['bRCCA'] = 0
optionsClassCCF['rccaLengthScale'] = 0.1000
optionsClassCCF['rccaNFeatures'] = 50
optionsClassCCF['rccaRegLambda'] = 1.0000e-03
optionsClassCCF['rccaIncludeOriginal'] = 0
optionsClassCCF['classNames'] = np.array([])
optionsClassCCF['org_muY']    = np.array([])
optionsClassCCF['org_stdY']   = np.array([])
optionsClassCCF['mseTotal']   = np.array([])
optionsClassCCF['task_ids']   = np.array([])

# print(optionsClassCCF)


# Load data
Tdata  = scipy.io.loadmat('./data_test/data.mat')
XTrain = Tdata['XTrain']
YTrain = Tdata['YTrain']
XTest  = Tdata['XTest']
YTest  = Tdata['YTest']
print(XTrain.shape)
print(YTrain.shape)

# Call CCF
CCF = genCCF(XTrain, YTrain, nTrees=200, optionsFor=optionsClassCCF)
YpredCCF, _, _ = predictFromCCF(CCF, XTest)
print('CCF Test missclassification rate (lower better): ', (100*(1- np.mean(YTest==(YpredCCF), axis=0))),  '%')

# Plotting
plotCCFDecisionSurface(CCF, XTrain, X=XTest, Y=YTest)




# XTrain1 = XTrain[0:500, :]
# YTrain1 = YTrain[0:500, :]
# XTest1  = XTrain[109:120, :]
# YTest1  = YTrain[109:120, :]
#
# CCF = genCCF(XTrain1, YTrain1,  nTrees=100, optionsFor=optionsClassCCF)
# YpredCCF, _, _ = predictFromCCF(CCF, XTest1)
#
# print('CCF Test missclassification rate (lower better): ', (100*(1- np.mean(YTest1==(YpredCCF)))),  '%')
