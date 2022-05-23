import copy
from typing import List, Optional, Tuple, NamedTuple, Union, Callable

import torch
from torch import Tensor
from torch_sparse import SparseTensor
import torch.nn as nn
from tqdm import tqdm
from torch_geometric.data import Data
import numpy as np
    
    
def k_hop_subgraph(node_idx, num_hops, edge_index, relabel_nodes=False,
                   num_nodes=None, flow='source_to_target'):
    r"""Computes the :math:`k`-hop subgraph of :obj:`edge_index` around node
    :attr:`node_idx`.
    It returns (1) the nodes involved in the subgraph, (2) the filtered
    :obj:`edge_index` connectivity, (3) the mapping from node indices in
    :obj:`node_idx` to their new location, and (4) the edge mask indicating
    which edges were preserved.

    Args:
        node_idx (int, list, tuple or :obj:`torch.Tensor`): The central
            node(s).
        num_hops: (int): The number of hops :math:`k`.
        edge_index (LongTensor): The edge indices.
        relabel_nodes (bool, optional): If set to :obj:`True`, the resulting
            :obj:`edge_index` will be relabeled to hold consecutive indices
            starting from zero. (default: :obj:`False`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        flow (string, optional): The flow direction of :math:`k`-hop
            aggregation (:obj:`"source_to_target"` or
            :obj:`"target_to_source"`). (default: :obj:`"source_to_target"`)

    :rtype: (:class:`LongTensor`, :class:`LongTensor`, :class:`LongTensor`,
             :class:`BoolTensor`)
    """

    num_nodes = num_nodes

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]

    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        subsets.append(col[edge_mask])

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return [subset, edge_index, inv, edge_mask]



class subgraph_extractor(nn.Module):
    # Our model

    def __init__(self, num_hop,k,edge_index,edge_value,total_num_nodes,device):
        super(subgraph_extractor, self).__init__()
        self.num_hop = num_hop
        self.k = k
        self.edge_index = edge_index
        self.edge_value = edge_value
        self.total_num_nodes = total_num_nodes
        self.device = device
        
    def forward(self, node_list):
         # Take too many times. Save the structure.

        sub_graph_list = []
        num_edges = []


        for i in tqdm(node_list):

            [tmp1,tmp2,tmp3,tmp4] = k_hop_subgraph([i],self.num_hop,self.edge_index,num_nodes=self.total_num_nodes,relabel_nodes=True)
            x = tmp1 
            edge_index_each = tmp2[:,:self.k]
            edge_value_masked = self.edge_value*tmp4
            edge_attr = edge_value_masked[edge_value_masked.nonzero(as_tuple=True)]
            edge_attr = edge_attr[:self.k]

            node_position = torch.LongTensor([tmp3]).to(self.device)
            y = node_position
            graph_each = Data(x=x, edge_index=edge_index_each, edge_attr=edge_attr, y = node_position)
#             graph_each = (x.to(self.device),y.to(self.device),edge_attr.to(self.device),edge_index_each.to(self.device))
            sub_graph_list.append(graph_each)

            num_edges.append(edge_index_each.shape[1])


        print("Average subgraph edges %.2f" % np.mean(num_edges))
        print("Num of no edge %d" % np.sum(num_edges == 0))

        return sub_graph_list
    

