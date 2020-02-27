# Calculates number of nodes in a tree

def get_number_of_nodes(tree):
    if tree["bLeaf"]:
        n_nodes = 1
    else:
        n_nodes_left  = get_number_of_nodes(tree["lessthanChild"])
        n_nodes_right = get_number_of_nodes(tree["greaterthanChild"])
        n_nodes = n_nodes_left + n_nodes_right + 1

    return n_nodes
