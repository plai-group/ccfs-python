import inspect
import numpy as np
import numpy.matlib as npmat

def traverse_tree_predict(tree, X):
    """
    Traverses the tree to get a prediction.  Splits X to left and right child
    then recursively calls self using each partition and the corresponding
    left and right sub tree.  This continues until called on a leaf, where it
    returns the mean of the leaf and, if requested, the full details of the
    corresponding leaf node.  These are then returned as array with the same
    number of rows as X.
    """
    if tree["bLeaf"]:
        leaf_mean = np.multiply(tree["mean"], np.ones((X.shape[0], 1)))
        leaf_node = npmat.repmat(tree, X.shape[0], 1)

    else:
        if ('rotDetails' in tree.keys()):
            if not (len(tree["rotDetails"]) == 0):
                X = np.dot(np.subtract(X, tree["rotDetails"]["muX"]), tree["rotDetails"]["R"])

        if ('featureExpansion' in tree.keys()):
            if inspect.isfunction(tree["featureExpansion"]):
                bLessChild = np.dot(tree["featureExpansion"](X[:, tree["iIn"]]), tree["decisionProjection"]) <= tree["paritionPoint"]
            else:
                bLessChild = np.dot((X[:, tree["iIn"]]), tree["decisionProjection"]) <= tree["paritionPoint"]
        else:
            if len(tree["decisionProjection"].shape) < 2:
                decisionProjection = np.expand_dims(tree["decisionProjection"], axis=1)
            else:
                decisionProjection = tree["decisionProjection"]
            
            bLessChild = np.dot(X[:, tree["iIn"]], decisionProjection) <= tree["paritionPoint"]

        leaf_mean =  np.empty((X.shape[0], tree["mean"].size))
        leaf_mean.fill(np.nan)
        node = np.array([{}])
        leaf_node = npmat.repmat(node, X.shape[0], 1)

        if len(bLessChild.shape) > 1:
            if bLessChild.shape[1] == 1:
                bLessChild = np.squeeze(bLessChild, axis=1)
            else:
                bLessChild = np.squeeze(bLessChild, axis=0)

        if np.any(bLessChild):
            leaf_mean[bLessChild, :], leaf_node[bLessChild] = traverse_tree_predict(tree["lessthanChild"], X[bLessChild, :])

        if np.any(~bLessChild):
            leaf_mean[~bLessChild, :], leaf_node[~bLessChild] = traverse_tree_predict(tree["greaterthanChild"], X[~bLessChild, :])

    return leaf_mean, leaf_node
