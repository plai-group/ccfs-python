
def  traverse_tree_predict(tree, X):
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

        if nargout>1
            leaf_node = repmat({tree},size(X,1),1);
        end

    else:
        if ('rotDetails' in tree) and (not tree["rotDetails"].size == 0):
            X = np.subtract(X, tree["rotDetails"]["muX"]) * tree["rotDetails"]["R"]
    
    return leaf_mean, leaf_node
