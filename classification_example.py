import scipy.io
import numpy as np
from collections  import OrderedDict
from src.generate_CCF import genCCF
from src.predict_from_CCF import predictFromCCF
from src.plotting.plot_surface import plotCCFDecisionSurface

# Sample Spital Data Testing script

# ----------------------Use optionsClassCCF---------------------------------#
# Default HyperParams
optionsClassCCF = {}
optionsClassCCF['lambda']           = 'log'
optionsClassCCF['splitCriterion']   = 'info'
optionsClassCCF['minPointsLeaf']    = 1
optionsClassCCF['bUseParallel']     = 1
optionsClassCCF['bCalcTimingStats'] = 1
optionsClassCCF['bSepPred']         = False
optionsClassCCF['taskWeights']      = 'even'
optionsClassCCF['bProjBoot']        = 'default'
optionsClassCCF['bBagTrees']        = 'default'
optionsClassCCF['projections']      = OrderedDict() # To ensure consistent order
optionsClassCCF['projections']['CCA'] = True
optionsClassCCF['treeRotation']       = None
optionsClassCCF['propTrain']          = 1
optionsClassCCF['epsilonCCA']         = 1.0000e-04
optionsClassCCF['mseErrorTolerance']  = 1.0000e-06
optionsClassCCF['maxDepthSplit'] = 'stack'
optionsClassCCF['XVariationTol'] = 1.0e-10
optionsClassCCF['RotForM']  = 3
optionsClassCCF['RotForpS'] = 0.7500
optionsClassCCF['RotForpClassLeaveOut'] = 0.5000
optionsClassCCF['minPointsForSplit']    = 2
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

#-------------------------------------------------------------------------------
# Load data
Tdata  = scipy.io.loadmat('/ccfs-python/dataset/spiral.mat')
XTrain = Tdata['XTrain']
YTrain = Tdata['YTrain']
XTest  = Tdata['XTest']
YTest  = Tdata['YTest']
print('Dataset Loaded!')

# Call CCF
print('CCF.......')
CCF = genCCF(XTrain, YTrain, nTrees=200, optionsFor=optionsClassCCF)
YpredCCF, _, _ = predictFromCCF(CCF, XTest)
print('CCF Test missclassification rate (lower better): ', (100*(1- np.mean(YTest==(YpredCCF), axis=0))),  '%')

#-------------------------------------------------------------------------------
# Plotting
x1Lims = [np.round(np.min(XTrain[:, 0])-1), np.round(np.max(XTrain[:, 0])+1)]
x2Lims = [np.round(np.min(XTrain[:, 1])-1), np.round(np.max(XTrain[:, 1])+1)]

plotCCFDecisionSurface("spiral_contour.svg", CCF, x1Lims, x2Lims, XTrain, X=XTest, Y=YTest)
