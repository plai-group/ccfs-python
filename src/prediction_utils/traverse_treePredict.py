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
        #print('not using')
        leaf_mean = np.multiply(tree["mean"], np.ones((X.shape[0], 1)))
        leaf_node = npmat.repmat(tree, X.shape[0], 1)

    else:
        #print('using')
        if ('rotDetails' in tree.keys()):
            if not (tree["rotDetails"].size == 0):
                X = np.subtract(X, tree["rotDetails"]["muX"]) @ tree["rotDetails"]["R"]

        if ('featureExpansion' in tree.keys()):
            if not (len(tree["featureExpansion"]) == 0):
                #print('not featureExpansion')
                #print((tree["featureExpansion"](X[:, tree["iIn"]]).shape))
                #print(tree["decisionProjection"].shape)
                #print(tree["paritionPoint"].shape)
                #print('^^^^^^^^^^^^^^^^^^^^^')
                bLessChild = np.dot(tree["featureExpansion"](X[:, tree["iIn"]]), tree["decisionProjection"]) <= tree["paritionPoint"]
            else:
                #print('not featureExpansion')
                #print(X[:, tree["iIn"]].shape)
                #print(tree["decisionProjection"].shape)
                #print(tree["paritionPoint"].shape)
                #print('^^^^^^^^^^^^^^^^^^^^^')
                bLessChild = np.dot((X[:, tree["iIn"]]), tree["decisionProjection"]) <= tree["paritionPoint"]
        else:
            #print('usecase-2')
            #print(tree["decisionProjection"].shape)
            #print(tree["paritionPoint"].shape)
            #print(X[:, tree["iIn"]].shape)
            #print('%%%%%%%%%%%%-2')
            bLessChild = np.dot((X[:, tree["iIn"]]), tree["decisionProjection"]) <= tree["paritionPoint"]

        #print(bLessChild)
        leaf_mean =  np.empty((X.shape[0], tree["mean"].size))
        leaf_mean.fill(np.nan)
        node = np.array([{}])
        leaf_node = npmat.repmat(node, X.shape[0], 1)

        if np.any(bLessChild):
            leaf_mean[bLessChild, :], leaf_node[bLessChild] = traverse_tree_predict(tree["lessthanChild"], X[bLessChild, :])

        if np.any(~bLessChild):
            leaf_mean[~bLessChild, :], leaf_node[~bLessChild] = traverse_tree_predict(tree["greaterthanChild"], X[~bLessChild, :])

    return leaf_mean, leaf_node
