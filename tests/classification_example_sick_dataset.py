import scipy.io
import numpy as np
import pandas as pd
from collections  import OrderedDict
from src.generate_CCF import genCCF
from sklearn.impute import SimpleImputer
from src.predict_from_CCF import predictFromCCF
from src.plotting.plot_surface import plotCCFClfyDecisionSurface
# SKlearn
from sklearn.metrics import f1_score

# Sample Script for testing sick dataset

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
## TRAIN
# Load data
XTrain = pd.read_csv('../dataset/sick.csv')
YTrain = pd.DataFrame(XTrain.pop('Class'))

# Basic missing values imputation
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
XTrain = imputer.fit_transform(XTrain)
XTrain = pd.DataFrame(XTrain)

# Make sure X, y are in pandas DataFrame
if isinstance(XTrain, pd.DataFrame):
    print(XTrain)
if isinstance(YTrain, pd.DataFrame):
    print(YTrain)

print('Dataset Loaded!')

# Call CCF
print('CCF.......')
CCF = genCCF(XTrain, YTrain, nTrees=100, optionsFor=optionsClassCCF, do_parallel=True)

## TEST
# Load data
XTest = pd.read_csv('../dataset/sick_test.csv')
YTest = pd.DataFrame(XTest.pop('Class'))

# Basic missing values imputation
XTest = imputer.fit_transform(XTest)
XTest = pd.DataFrame(XTest)

# Make sure X, y are in pandas DataFrame
if isinstance(XTest, pd.DataFrame):
    print(XTest)
if isinstance(YTest, pd.DataFrame):
    print(YTest)

# Prediction
YpredCCF, _, _ = predictFromCCF(CCF, XTest)

# Evaluate
## Accuracy
print('CCF Test missclassification rate (lower better): ', (100*(1- np.mean(YTest.to_numpy()==YpredCCF, axis=0))),  '%')

## F1-Score
score = f1_score(YTest, YpredCCF, pos_label='sick')
print('F1 Score: ', (score))
#-------------------------------------------------------------------------------

