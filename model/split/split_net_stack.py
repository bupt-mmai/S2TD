"""DFS-L version 
change line 114 & 145 & 146 &
       line 194 & 230 & 231
"""
import torch
from torch import nn
from treelib import Tree

from model.split.attention import AdditiveAttention, ScaledDotAttention


class SplitNet(nn.Module):

    def __init__(self, input_size, max_len, att_type, threshold=0.3):
        """

        :param input_size: (int)
        :param max_len:  (int)
        :param att_type: (str)
        :param threshold: (float)
        """

        super(SplitNet, self).__init__()

        assert att_type in {'add', 'scaled_dot'}

        self.input_size = input_size
        self.max_len = max_len
        self.threshold = threshold

        # === Not used, still conducting experiments on them ====
        if att_type == 'add':
            self.split_attention = AdditiveAttention(input_size, input_size, input_size)
        else:
            self.split_attention = ScaledDotAttention(input_size, input_size)
        # ===================================================

        self.split_gate = nn.Sequential(
            nn.LayerNorm(input_size),
            nn.Linear(input_size, input_size),
            nn.Sigmoid()
        )

    def split_node(self, parent_node, features, node_depth):
        """SpiltModule

        :param parent_node: (tensor) with shape (batch_size, input_size)
        :param features: (tensor) with shape (batch_size, f_max, input_size)
        :param node_depth: (tensor) with shape (batch_size,)
        :return: new left and right node's tensor with the same shape of input node
        """

        gate = self.split_gate(parent_node)

        left_node = gate * parent_node
        right_node = (1 - gate) * parent_node

        return left_node, right_node

    def filter_leaf_tensor(self, tree_tensor, tree_list, max_leaf_num):
        """

        :param tree_tensor: (tensor) shape is (batch_size, 2*max_len-1, input_size)
        :param tree_list: (list(Tree)) length of the list is batch_size
        :param max_leaf_num: (int) indicate the maximum number of leaves per tree
        :return: (batch_size, leaf_num, input_size)
        """
        assert isinstance(max_leaf_num, int) and max_leaf_num > 0

        batch_size = tree_tensor.shape[0]
        device = tree_tensor.device

        leaf_matrix = torch.zeros(batch_size, max_leaf_num, self.input_size).to(device)

        for i in range(batch_size):
            leaf_idx = torch.tensor([tree_list[i][node].data for node in tree_list[i].expand_tree(mode=Tree.DEPTH)
                                     if tree_list[i][node].is_leaf()][:max_leaf_num], dtype=torch.long)
            leaf_matrix[i, torch.arange(leaf_idx.shape[0]), :] = tree_tensor[i, leaf_idx, :]

        return leaf_matrix

    def forward_train(self, input_, features, length, label, return_tree_tensor=False):

        batch_size = input_.shape[0]
        device = input_.device

        sorted_length, index = (2 * length - 1).sort(descending=True)
        sorted_length = sorted_length.tolist()
        sorted_input = input_[index]
        sorted_label = label[index]
        sorted_features = features[index]

        # setup tree structure / add root node
        sorted_tree_list = [Tree() for i in range(batch_size)]
        for i, tree in zip(index, sorted_tree_list):
            tree.create_node(tag=i.item(), identifier=0, data=0)

        # use stack to store non-spilt nodes
        sorted_node_stack = [[(0, 0)] for i in range(batch_size)]  # (index, depth) 0-indexed

        # sorted_tree_tensor is a tensor keep track all the node tensors
        sorted_tree_tensor = torch.zeros(batch_size, 2*self.max_len-1, self.input_size, dtype=input_.dtype).to(device)
        sorted_tree_tensor[:, 0] = sorted_input
        sorted_scores = torch.zeros(batch_size, 2 * self.max_len - 1).to(device)

        new_left_node_idx = torch.ones(batch_size, dtype=torch.long)

        for i in range(max(sorted_length)):

            batch_size_i = sum([l > i for l in sorted_length])

            # get next node idx from the stack
            next_node_to_split = list()
            node_depth = list()  # depth of next_node_to_split in tree
            for si in range(batch_size_i):
                nidx, depth = sorted_node_stack[si].pop()  # here is DFS-like, if you want to use BFS-like, change pop() --> pop(0)
                next_node_to_split.append(nidx)
                node_depth.append(depth)
            next_node_to_split = torch.tensor(next_node_to_split, dtype=torch.long)
            node_depth = torch.tensor(node_depth, dtype=torch.long)

            parent_nodes = sorted_tree_tensor[torch.arange(batch_size_i), next_node_to_split, :]

            # split node into two new nodes
            left_nodes, right_nodes = self.split_node(parent_nodes, sorted_features[:batch_size_i], node_depth)

            # Score Module: calculate confidence of the potential split
            s = torch.cosine_similarity(left_nodes, right_nodes, dim=-1)  # (batch_size_i, )
            sorted_scores[:batch_size_i, i] = s

            # update tree structure based on ground truth labels
            keep = (sorted_label[:batch_size_i, i] > 0)

            sorted_tree_tensor[torch.arange(batch_size_i)[keep], new_left_node_idx[:batch_size_i][keep], :] = \
                left_nodes[keep]
            sorted_tree_tensor[torch.arange(batch_size_i)[keep], (new_left_node_idx + 1)[:batch_size_i][keep], :] = \
                right_nodes[keep]

            for ti, pn, ln, d in zip(torch.arange(batch_size_i)[keep].tolist(), next_node_to_split[keep].tolist(),
                                     new_left_node_idx[:batch_size_i][keep].tolist(), node_depth[keep].tolist()):
                # update tree
                sorted_tree_list[ti].create_node(tag=ln, identifier=ln, data=ln, parent=pn)
                sorted_tree_list[ti].create_node(tag=ln + 1, identifier=ln + 1, data=ln + 1, parent=pn)

                # update stack - right -> left
                # change the order can obtain different decoding strategy
                sorted_node_stack[ti].append((ln + 1, d + 1))
                sorted_node_stack[ti].append((ln, d + 1))

            new_left_node_idx[:batch_size_i][keep] += 2

        # transform back
        _, reverse_index = index.sort()
        tree_list = sorted(sorted_tree_list, key=lambda t: t[0].tag)  # sort tree list by root's tag
        scores = sorted_scores[reverse_index]
        tree_tensor = sorted_tree_tensor[reverse_index]

        # filter out non-leaf tensors
        leaf_tensor = self.filter_leaf_tensor(tree_tensor, tree_list, self.max_len)

        if return_tree_tensor:
            return leaf_tensor, tree_list, scores, tree_tensor
        else:
            return leaf_tensor, tree_list, scores

    def forward_test(self, input_, features, return_tree_tensor=False):

        batch_size = input_.shape[0]
        device = input_.device

        # setup tree structure / add root node
        tree_list = [Tree() for i in range(batch_size)]
        for i, tree in enumerate(tree_list):
            tree.create_node(tag=i, identifier=0, data=0)

        # use stack to store non-spilt nodes
        node_stack = [[(0, 0)] for i in range(batch_size)]

        # tree_tensor is a tensor keep track all the node tensors
        tree_tensor = torch.zeros(batch_size, 2 * self.max_len - 1, self.input_size, dtype=input_.dtype).to(device)
        tree_tensor[:, 0] = input_
        scores = torch.zeros(batch_size, 2 * self.max_len - 1).to(device)

        new_left_node_idx = torch.ones(batch_size, dtype=torch.long)

        for i in range(2 * self.max_len - 1):

            # get next node idx from the stack
            next_node_to_split = list()
            node_depth = list()
            ongoing = list()
            for si in range(batch_size):
                if node_stack[si] and new_left_node_idx[si] < 2*self.max_len-1:  # if not empty & not reach max leaf num
                    nidx, depth = node_stack[si].pop()  # here is DFS-like, if you want to use BFS-like, change pop() --> pop(0)
                    next_node_to_split.append(nidx)
                    node_depth.append(depth)
                    ongoing.append(si)

            if not ongoing:
                break
            next_node_to_split = torch.tensor(next_node_to_split, dtype=torch.long)
            node_depth = torch.tensor(node_depth, dtype=torch.long)
            ongoing = torch.tensor(ongoing, dtype=torch.long)

            parent_nodes = tree_tensor[ongoing, next_node_to_split, :]

            # split node into two new nodes
            left_nodes, right_nodes = self.split_node(parent_nodes, features[ongoing], node_depth)

            # Score Module: calculate confidence of the potential split
            s = torch.cosine_similarity(left_nodes, right_nodes, dim=-1)  # (batch_size_i, )
            scores[ongoing, i] = s

            # update tree structure based on ground truth labels
            keep = s <= self.threshold
            if not keep.any():
                continue

            tree_tensor[ongoing[keep], new_left_node_idx[ongoing[keep]], :] = left_nodes[keep]
            tree_tensor[ongoing[keep], (new_left_node_idx + 1)[ongoing[keep]], :] = right_nodes[keep]

            for ti, pn, ln, d in zip(ongoing[keep].tolist(), next_node_to_split[keep].tolist(),
                                  new_left_node_idx[ongoing[keep]].tolist(), node_depth[keep].tolist()):
                # update tree
                tree_list[ti].create_node(tag=ln, identifier=ln, data=ln, parent=pn)
                tree_list[ti].create_node(tag=ln + 1, identifier=ln + 1, data=ln + 1, parent=pn)

                # update stack - right -> left
                # change the order can obtain different decoding strategy
                node_stack[ti].append((ln + 1, d + 1))
                node_stack[ti].append((ln, d + 1))

            new_left_node_idx[ongoing[keep]] += 2

        # filter out non-leaf tensors
        leaf_tensor = self.filter_leaf_tensor(tree_tensor, tree_list, self.max_len)

        if return_tree_tensor:
            return leaf_tensor, tree_list, scores, tree_tensor
        else:
            return leaf_tensor, tree_list, scores

    def forward(self, input_, features, length=None, label=None, return_tree_tensor=False):
        """

        :param input_: (batch_size, input_size)
        :param features: (batch_size, f_max, input_size)
        :param length: None or tensor with shape (batch_size, ), num of leaf
        :param label: None or tensor with shape (batch_size, 2*max_len-1), 0 - leaf \ 1 - non-leaf
        :param return_tree_tensor: (bool) whether to return all tree_tensor
        :return: leaf_tensor (tensor) (batch_size, max_len, input_size), tree_list (list[Tree]),
                 scores (tensor) (batch_size, 2*max_len-1)
        """

        if self.training:
            return self.forward_train(input_, features, length, label, return_tree_tensor)
        else:
            return self.forward_test(input_, features, return_tree_tensor)
