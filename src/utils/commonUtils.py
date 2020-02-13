import numpy as np

def cohenKappa(confusionMatrix):
    """
    Calculates Cohen's kappa from a confusion matrix.
    Parameters
    ----------
    confusionMatrix: Numpy array

    Returns
    -------
    float
        a floating point value (Cohen's kappa) of the confusion matrix
    """
    propPreds = np.sum(confusionMatrix, axis=0)
    propPreds = propPreds/np.sum(propPreds)

    propReal  = np.sum(confusionMatrix, axis=1)
    propReal  = propReal/np.sum(propReal)

    p_e = np.sum(propPreds * propReal)
    p_o = np.sum(np.diag(confusionMatrix))/np.sum(confusionMatrix)

    kappa = (p_o - p_e)/(1 - p_e)

    return kappa
