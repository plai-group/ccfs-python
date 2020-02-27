from utils.ccfUtils import random_missing_vals
from prediction_utils.replicate_input_process import replicateInputProcess

def predictFromCCT(tree, X):
    """
    predictFromCCT predicts output using trained tree

    Parameters
    ----------
    tree = output strcut from growTree
    X = processed input features

    Returns
    -------
    leaf_mean = Mean of outputs present at the leaf.
                For classification then this represents the class
                probability, for regression it is simply the output mean.
    leaf_node = The full leaf node details for the assigned point.
    """
    if 'inputProcessDetails' in tree.keys():
        X = replicateInputProcess(X, tree["inputProcessDetails"])

    # Any values left as NaN now need to be randomly assigned
    X = random_missing_vals(X)

    leaf_mean, leaf_node = traverse_tree_predict(tree, X)

    return leaf_mean, leaf_node
