import scipy.io
import numpy as np
from collections  import OrderedDict
from src.generate_CCF import genCCF
from src.predict_from_CCF import predictFromCCF
from src.plotting.plot_surface import plotCCFRegDecisionSurface

# Sample Camel6 Data Testing script

# ----------------------Use optionsClassCCF---------------------------------#
# Default HyperParams
optionsClassCCF = {}
optionsClassCCF['lambda']           = 'log'
optionsClassCCF['splitCriterion']   = 'mse'
optionsClassCCF['minPointsLeaf']    = 3
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
optionsClassCCF['minPointsForSplit']    = 6
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


#-------------------------------------------------------------------------------
# Load data
Tdata  = scipy.io.loadmat('/ccfs-python/dataset/camel6.mat')
XTrain = Tdata['XTrain']
YTrain = Tdata['YTrain']
XTest  = Tdata['XTest']
YTest  = Tdata['YTest']
print('Dataset Loaded!')

# Call CCF
print('CCF.......')
CCF = genCCF(XTrain, YTrain, nTrees=200, bReg=True, optionsFor=optionsClassCCF, do_parallel=True)
YpredCCF, _, _ = predictFromCCF(CCF, XTest)
print('CCF Mean squared error (lower better): ', (np.mean((YpredCCF - YTest)**2)))

#-------------------------------------------------------------------------------
# Plotting
x1Lims = [-1.15, 1.15]
x2Lims = [-1.75, 1.75]
plotCCFRegDecisionSurface("camel_contour.svg", CCF, x1Lims, x2Lims, XTrain, X=XTest, Y=YTest, plot_X=False)
