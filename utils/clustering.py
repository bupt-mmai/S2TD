import torch
from treelib import Tree


def cosine_distance(start_points, end_points):
    """

    :param start_points: (tensor) with shape (num_points, emb_size)
    :param end_points: (tensor) with shape (num_points, emb_size)
    :return: distance (tensor) with shape (num_points, )
    """

    return -1 * torch.cosine_similarity(start_points, end_points, dim=1)  # (num_points, )


def hierarchical_cluster_neighbours(input_, distance_func):
    """

    :param input_: (tensor) with shape (length, emb_size)
    :param distance_func: (function) a distance calculating function (array, array) -> float
    :return: list(tuple) (left_node_idx, right_node_idx, parent_node_idx)
    """

    length = input_.shape[0]
    assert length >= 2, "invalid length of input"

    all_nodes = torch.zeros(2*length-1, input_.shape[1], dtype=input_.dtype)
    all_nodes[:length] = input_

    l2t_l = lambda l: torch.tensor(l, dtype=torch.long)  # list to long tensor
    l2t_f = lambda l: torch.tensor(l, dtype=torch.float32)  # list to float tensor

    leaf_index = list(range(length))  # [0, 1, 2, ..., length-1]
    distance = distance_func(all_nodes[l2t_l(leaf_index[:-1])], all_nodes[l2t_l(leaf_index[1:])])  # tensor (length-1, )

    cidx = torch.argmin(distance).item()
    distance = distance.tolist()

    results = list()

    for nidx in range(length, 2*length-1): # next cluster node

        results.append((leaf_index[cidx], leaf_index[cidx+1], nidx))

        # acquire representation of the new node
        all_nodes[nidx] = (all_nodes[leaf_index[cidx]] + all_nodes[leaf_index[cidx+1]]) / 2

        leaf_index = leaf_index[:cidx] + [nidx] + leaf_index[cidx+2:]

        if len(leaf_index) == 1:
            break

        updated_index = leaf_index[max(cidx-1, 0): cidx+2]
        updated_distance = distance_func(all_nodes[l2t_l(updated_index[:-1])], all_nodes[l2t_l(updated_index[1:])])

        distance = distance[:max(cidx-1, 0)] + updated_distance.tolist() + distance[cidx+2:]

        cidx = torch.argmin(l2t_f(distance)).item()

    return results


def cluster_results_to_tree(results):
    """Right way to show tree structure: t.show(key=lambda n: n.data.order)

    :param results: list(tuple) (left_node_idx, right_node_idx, parent_node_idx)
    :return: a Tree instance
    """

    t = Tree()

    class NodeData(object):

        def __init__(self, order):
            self.order = order

    order = 0

    if len(results) == 0:
        t.create_node(tag='0', identifier=0, data=NodeData(order))
        return t

    for r in reversed(results):

        if t.root is None:
            t.create_node(tag=str(r[-1]), identifier=r[-1], data=NodeData(order))
            order += 1

        t.create_node(tag=str(r[0]), identifier=r[0], parent=r[-1], data=NodeData(order))
        order += 1
        t.create_node(tag=str(r[1]), identifier=r[1], parent=r[-1], data=NodeData(order))
        order += 1

    return t


def cluster_results_to_labels(results):
    """

    :param results: list(tuple) (left_node_idx, right_node_idx, parent_node_idx)
    :return: list of labels
    """

    labels = list()
    if not results:
        return labels

    nodes = [results[-1][-1]]
    leaf_num = len(results)

    while nodes:

        n = nodes.pop()

        if n > leaf_num: # non-leaf
            labels.append(1)
            nodes.append(results[n-leaf_num-1][1])  # right
            nodes.append(results[n-leaf_num-1][0])  # left
        else:
            labels.append(0)

    assert not labels or len(labels) == results[-1][-1] + 1, 'invalid labels'

    return labels


def cluster_results_to_scores(results, c=0.2):
    """

    :param results: list(tuple) (left_node_idx, right_node_idx, parent_node_idx)
    :param c: (float)
    :return: list of scores
    """

    t = cluster_results_to_tree(results)
    return [c * t.depth(n) for n in t.expand_tree(mode=Tree.DEPTH, key=lambda n: n.data.order)]
