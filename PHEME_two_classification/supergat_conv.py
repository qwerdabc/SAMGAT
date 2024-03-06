from collections import deque
import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
import numpy as np
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import OptTensor
from torch_geometric.utils import (
    add_self_loops,
    batched_negative_sampling,
    dropout_adj,
    is_undirected,
    negative_sampling,
    remove_self_loops,
    softmax,
    to_undirected,
    to_dense_adj
)
from torch_scatter import scatter_add,scatter_max
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn import GCNConv

from torch_geometric.utils.num_nodes import maybe_num_nodes
def topk_softmax(src, index, k, num_nodes=None):
    r"""Computes a sparsely evaluated softmax using only top-k values.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then select top-k values for each group, compute the softmax individually.
    The output of not selected indices will be zero.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor): The indices of elements for applying the softmax.
        k (int): The number of indexes to select from each group.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)

    :rtype: :class:`Tensor`
    """
    num_nodes = maybe_num_nodes(index, num_nodes)

    # Create mask the topk values of which are 1., otherwise 0.
    out = src.clone()
    topk_mask = torch.zeros_like(out)
    for _ in range(k):
        v_max = scatter_max(out, index, dim=0, dim_size=num_nodes)[0]
        i_max = (out == v_max[index])
        topk_mask[i_max] = 1.
        out[i_max] = float("-Inf")

    out = src - scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
    out = out.exp()

    # Mask except topk values
    out = out * topk_mask

    out = out / (
        scatter_add(out, index, dim=0, dim_size=num_nodes)[index] + 1e-16)
    return out

def similarity_select_n(x, new_edge_index, n):
    x = torch.tensor(x, dtype=torch.float32)
    normalized_x = x / x.norm(dim=-1, keepdim=True)
    
    # Calculate the similarity scores for the edges in new_edge_index
    similarity_scores = [torch.dot(normalized_x[u], normalized_x[v]).item() for u, v in new_edge_index.transpose().tolist()]
    
    # Sort the scores and indices 
    sorted_indices_scores = sorted(enumerate(similarity_scores), key=lambda x: x[1], reverse=True)
    
    # Keep track of how many edges have been added for each node
    node_edge_counts = {}
    
    # The indices of the edges to keep
    keep_indices = []
    
    # Loop over the sorted (index, score) tuples
    for index, score in sorted_indices_scores:
        # Get the nodes this edge is connected to
        u, v = new_edge_index.transpose()[index]
        
        # If adding this edge does not cause either node to exceed the limit
        if node_edge_counts.get(u, 0) < n and node_edge_counts.get(v, 0) < n:
            # Add this edge
            keep_indices.append(index)
            
            # Update the edge counts
            node_edge_counts[u] = node_edge_counts.get(u, 0) + 1
            node_edge_counts[v] = node_edge_counts.get(v, 0) + 1
            
        # If we have already added n edges for each node, we can stop early
        if len(keep_indices) == n:
            break
            
    # Extract the edges to keep
    ratio_edge_index = new_edge_index.T[keep_indices].T
    return ratio_edge_index
def similarity_select(x,new_edge_index,percentage):
    x = torch.tensor(x,dtype=torch.float32)
    normalized_x = x / x.norm(dim=-1, keepdim=True)

    # 计算 new_edge_index 对应的边的特征相似度
    similarity_scores = [torch.dot(normalized_x[u], normalized_x[v]).item() for u, v in new_edge_index.transpose().tolist()]
    #print(similarity_scores)
    # 对相似度进行排序并选取前百分比的边
    num_top_edges = int(len(similarity_scores) * percentage)
    sorted_indices = sorted(range(len(similarity_scores)), key=lambda i: similarity_scores[i], reverse=True)[:num_top_edges]
    #print(sorted_indices)
    # 提取 ratio_edge_index
    ratio_edge_index = new_edge_index.T[sorted_indices].T
    return ratio_edge_index
def sibling_edge(x,edge_index,ratio,sta=True,limitn=False):
    edge_index = edge_index.cpu().numpy()
    x = x.cpu().detach().numpy()
    def bfs(edge_index, root):
        visited = set([root])
        layers = {0: [root]}
        parent_map = {root: None}
        queue = deque([(root, 0)])

        while queue:
            node, level = queue.popleft()
            next_level = level + 1
            if next_level not in layers:
                layers[next_level] = []

            for i in range(edge_index.shape[1]):
                if edge_index[0, i] == node:
                    neighbor = edge_index[1, i]
                    if neighbor not in visited:
                        visited.add(neighbor)
                        layers[next_level].append(neighbor)
                        parent_map[neighbor] = node
                        queue.append((neighbor, next_level))

        return layers, parent_map
    root = set(edge_index[0]) - set(edge_index[1])
    # if len(root) != 32:
    #     raise ValueError("more than 1 root",root)
    if sta is True:
        updated_edge_index = edge_index
    else:
        updated_edge_index = np.array([[],[]])
    for i in range(len(root)):
        rt = root.pop()
        layers, parent_map = bfs(edge_index, rt)

        # 添加具有相同父节点的兄弟节点之间的边（无向图）
        new_edges = []
        for layer_nodes in layers.values():
            for i in range(len(layer_nodes)):
                for j in range(i + 1, len(layer_nodes)):
                    if parent_map[layer_nodes[i]] == parent_map[layer_nodes[j]]:
                        new_edges.append([layer_nodes[i], layer_nodes[j]])
                        new_edges.append([layer_nodes[j], layer_nodes[i]])
        #print(edge_index,new_edges)
        new_edge_index = np.array(new_edges).T
        if len(new_edge_index) != 0:
            if limitn is False:
                new_edge_index = similarity_select(x,new_edge_index,ratio)
            else:
                new_edge_index = similarity_select_n(x,new_edge_index,1)
            #new_edge_index = random_select(new_edge_index,ratio)
            updated_edge_index = np.hstack([updated_edge_index,new_edge_index])
    return updated_edge_index


class SuperGATConv(MessagePassing):
    r"""The self-supervised graph attentional operator from the `"How to Find
    Your Friendly Neighborhood: Graph Attention Design with Self-Supervision"
    <https://openreview.net/forum?id=Wi5KUNlqWty>`_ paper

    .. math::

        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},

    where the two types of attention :math:`\alpha_{i,j}^{\mathrm{MX\ or\ SD}}`
    are computed as:

    .. math::

        \alpha_{i,j}^{\mathrm{MX\ or\ SD}} &=
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(
            e_{i,j}^{\mathrm{MX\ or\ SD}}
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(
            e_{i,k}^{\mathrm{MX\ or\ SD}}
        \right)\right)}

        e_{i,j}^{\mathrm{MX}} &= \mathbf{a}^{\top}
            [\mathbf{\Theta}\mathbf{x}_i \, \Vert \,
             \mathbf{\Theta}\mathbf{x}_j]
            \cdot \sigma \left(
                \left( \mathbf{\Theta}\mathbf{x}_i \right)^{\top}
                \mathbf{\Theta}\mathbf{x}_j
            \right)

        e_{i,j}^{\mathrm{SD}} &= \frac{
            \left( \mathbf{\Theta}\mathbf{x}_i \right)^{\top}
            \mathbf{\Theta}\mathbf{x}_j
        }{ \sqrt{d} }

    The self-supervised task is a link prediction using the attention values
    as input to predict the likelihood :math:`\phi_{i,j}^{\mathrm{MX\ or\ SD}}`
    that an edge exists between nodes:

    .. math::

        \phi_{i,j}^{\mathrm{MX}} &= \sigma \left(
            \left( \mathbf{\Theta}\mathbf{x}_i \right)^{\top}
            \mathbf{\Theta}\mathbf{x}_j
        \right)

        \phi_{i,j}^{\mathrm{SD}} &= \sigma \left(
            \frac{
                \left( \mathbf{\Theta}\mathbf{x}_i \right)^{\top}
                \mathbf{\Theta}\mathbf{x}_j
            }{ \sqrt{d} }
        \right)

    .. note::

        For an example of using SuperGAT, see `examples/super_gat.py
        <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
        super_gat.py>`_.

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        attention_type (string, optional): Type of attention to use.
            (:obj:`'MX'`, :obj:`'SD'`). (default: :obj:`'MX'`)
        neg_sample_ratio (float, optional): The ratio of the number of sampled
            negative edges to the number of positive edges.
            (default: :obj:`0.5`)
        edge_sample_ratio (float, optional): The ratio of samples to use for
            training among the number of training edges. (default: :obj:`1.0`)
        is_undirected (bool, optional): Whether the input graph is undirected.
            If not given, will be automatically computed with the input graph
            when negative sampling is performed. (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          negative edge indices :math:`(2, |\mathcal{E}^{(-)}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, H * F_{out})`
    """
    att_x: OptTensor
    att_y: OptTensor
    att_xs: OptTensor
    att_ys: OptTensor

    def __init__(self, in_channels: int, out_channels: int, heads: int = 1,
                 concat: bool = True, negative_slope: float = 0.2,
                 dropout: float = 0.0, add_self_loops: bool = True,
                 bias: bool = True, attention_type: str = 'MX',
                 neg_sample_ratio: float = 0.5, edge_sample_ratio: float = 1.0,
                 is_undirected: bool = False,mode: str = "additive",residual: bool =False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)
        self.mod = mode
        self.residual = residual
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w = Parameter(torch.ones(self.out_channels))
        self.l1 = Parameter(torch.FloatTensor(1, self.out_channels))
        self.b1 = Parameter(torch.FloatTensor(1, self.out_channels))
        self.l2 = Parameter(torch.FloatTensor(self.out_channels, self.out_channels))
        self.b2 = Parameter(torch.FloatTensor(1, self.out_channels))
        self.activation = torch.nn.ReLU()
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.attention_type = attention_type
        self.neg_sample_ratio = neg_sample_ratio
        self.edge_sample_ratio = edge_sample_ratio
        self.sibling_sample_ratio = 0.3
        self.is_undirected = is_undirected
        self.use_DropKey = False
        #self.feat_drop = False
        #self.feat_dropout = torch.nn.Dropout(0.3)

        assert attention_type in ['MX', 'SD', 'ADD']
        assert 0.0 < neg_sample_ratio and 0.0 < edge_sample_ratio <= 1.0

        self.lin = Linear(in_channels, heads * out_channels, bias=False,
                          weight_initializer='glorot')

        self.eps = Parameter(torch.Tensor([0.]))
        self.beta = Parameter(torch.Tensor([0.01]))
        self.rate = Parameter(torch.Tensor([0.5]))
        self.enable_norm = True
        self.use_topk_softmax = True
        #self.norm = torch.nn.BatchNorm1d(out_channels)#(heads * out_channels)
        #self.convattq = GCNConv(in_channels,out_channels,normalize=False)
        #self.convattk = GCNConv(out_channels,out_channels,normalize=False)
        #self.ww=torch.nn.Parameter(torch.ones(2)) 
        #self.sm = torch.nn.Softmax()
        
        #self.lin = FF(in_channels,heads * out_channels)
        # torch.nn.Sequential(Linear(in_channels, in_channels,bias=False,
        #                   weight_initializer='glorot'), torch.nn.Dropout(self.dropout), 
        #                       torch.nn.ReLU(), 
        #                       Linear(in_channels,  heads * out_channels,bias=False,
        #                   weight_initializer='glorot'))
        # #torch.nn.Sequential(Linear(out_channels, out_channels), torch.nn.ReLU(), Linear(out_channels, out_channels))
        

        if self.attention_type == 'MX':
            self.att_l = Parameter(torch.Tensor(1, heads, out_channels))
            self.att_r = Parameter(torch.Tensor(1, heads, out_channels))
        elif self.attention_type == 'SD':
            self.register_parameter('att_l', None)
            self.register_parameter('att_r', None)
        elif self.attention_type == 'ADD':
            self.att_l = Parameter(torch.Tensor(1, heads, out_channels))
            self.att_r = Parameter(torch.Tensor(1, heads, out_channels))

        self.att_x = self.att_y = self.att_xs = self.att_ys = None  # x/y for self-supervision

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        glorot(self.att_l)
        glorot(self.att_r)
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Tensor,sibling_index: OptTensor = None,sibling_indexn: OptTensor = None,oriindex:OptTensor = None, mlp: Optional[torch.nn.Module] = None,
                neg_edge_index: OptTensor = None,
                batch: OptTensor = None) -> Tensor:
        r"""
        Args:
            neg_edge_index (Tensor, optional): The negative edges to train
                against. If not given, uses negative sampling to calculate
                negative edges. (default: :obj:`None`)
        """
        # sibling_index=sibling_edge(x,oriindex,0.4,False,False)
        # sibling_index = torch.LongTensor(sibling_index).to(x.device)
        # sibling_indexn=sibling_edge(x,oriindex,0.4,False,True)
        # sibling_indexn = torch.LongTensor(sibling_indexn).to(x.device)
        N, H, C = x.size(0), self.heads, self.out_channels
        #edge_index = to_undirected(edge_index, num_nodes=x.size(0))
        if self.add_self_loops:
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=N)

        # Q=self.convattq(x,edge_index)
        # K=self.convattq(x,edge_index)
        #print(x.shape,Q.shape,K.shape,edge_index.shape)
        # if self.feat_drop:
        #     x = self.feat_dropout(x)
        if mlp != None:
            x = mlp(x).view(-1, H, C)#self.lin(x).view(-1, H, C)
        else:
            x = self.lin(x).view(-1, H, C)
            #x = x + (torch.randn(x.size()) * 0.01).to(self.eps.device)
        # W_att=self.sm(self.ww)
        #print(W_att.shape)
        #matmul(adj_t, x, reduce=self.aggr)
        #tmp_out_att=torch.mm(edge_index_i,x_i)
        # dense_m = to_dense_adj(edge_index,batch)
        # attention=self._soft_max_att(dense_m,torch.mm(Q,K.T))*W_att[0] \
        #     +dense_m*W_att[1]
        attention = torch.tensor(1)

        # propagate_type: (x: Tensor)
        #propa_edge_index = torch.hstack([edge_index,sibling_indexn])
        out = self.propagate(edge_index, x=x, size=None, num_nodes=x.size(0),attention=attention)
        
        
        #out = self.lin2(out)

        if self.training:
            #edge_index = torch.hstack([edge_index,sibling_index])
            pos_edge_index = self.positive_sampling(edge_index)
            if len(sibling_index.shape) != 1:
                print(sibling_index.shape())
                neu_sibling_index = self.sibling_sampling(sibling_index) #sibling_samping 0.1还没运行，现在运行的还是positive sampling
            else:
                neu_sibling_index = sibling_index
            #pos_edge_index = torch.hstack([pos_edge_index,pos_sibling_index])

            pos_att = self.get_attention(
                edge_index_i=pos_edge_index[1],
                x_i=x[pos_edge_index[1]],
                x_j=x[pos_edge_index[0]],
                num_nodes=x.size(0),
                x=x,attention=attention,
                edge_index=pos_edge_index,
                return_logits=True,
            )
            # neu_att = self.get_attention(
            #     edge_index_i=neu_sibling_index[1],
            #     x_i=x[neu_sibling_index[1]],
            #     x_j=x[neu_sibling_index[0]],
            #     num_nodes=x.size(0),
            #     x=x,attention=attention,
            #     edge_index=neu_sibling_index,
            #     return_logits=True,
            # )

            if neg_edge_index is None:
                #edge_index = torch.hstack([edge_index,sibling_index]) #之前没有这一行,neg把sibling也算进去了
                neg_edge_index = self.negative_sampling(edge_index, N, batch)

            neg_att = self.get_attention(
                edge_index_i=neg_edge_index[1],
                x_i=x[neg_edge_index[1]],
                x_j=x[neg_edge_index[0]],
                x=x,attention=attention,
                edge_index=neg_edge_index,
                num_nodes=x.size(0),
                return_logits=True,
            )

            self.att_x = torch.cat([pos_att, neg_att], dim=0)
            #self.att_x = torch.cat([pos_att, neg_att, neu_att], dim=0)
            #self.att_xs = neu_att
            self.att_y = self.att_x.new_zeros(self.att_x.size(0))
            self.att_y[:pos_edge_index.size(1)] = 1.
            #self.att_y[-neu_sibling_index.size(1):] = 0.4
            #self.att_ys = self.att_xs.new_zeros(self.att_xs.size(0))
            #self.att_ys[:] = 0.4
            

        # if self.enable_norm:
        #     out = out.view(-1, self.heads * self.out_channels)
        #     out = self.norm(out)
        #     out = out.view(-1, self.heads, self.out_channels)
        if self.concat is True:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
            #out = self.norm(out)
        
        

        if self.bias is not None:
            out += self.bias

        return out
    def top_k_attention(self, alpha: torch.Tensor, edge_index: torch.Tensor, num_nodes: int, k: int):
        """
        Selects the top-k attention values for each node and sets the rest to zero.
        """
        # Create a tensor to hold the new attention values
        new_alpha = torch.zeros_like(alpha)

        # For each node...
        for node in range(num_nodes):
            # Find the edges for this node
            edge_indices = (edge_index == node).nonzero(as_tuple=True)[0]

            # If there are no edges for this node, continue to the next node
            if edge_indices.numel() == 0:
                continue

            # Get the attention values for these edges
            node_alpha = alpha[edge_indices]

            # If there are more than k edges...
            if node_alpha.shape[0] > k:
                # Find the values and indices of the top-k attention values
                top_values, top_indices = torch.topk(node_alpha, k,dim=0, largest=True)

                # Create a mask of the same size as node_alpha, with True at the indices of the top-k values
                mask = torch.full(node_alpha.shape, False, dtype=torch.bool,device=node_alpha.device)
                col_indices = torch.arange(node_alpha.size(1)).unsqueeze(0).expand_as(top_indices)

                mask[top_indices, col_indices] = True

                # Set the attention values that are not in the top-k to zero
                node_alpha = node_alpha.masked_fill(~mask, 0)
            # If there are less or equal to k edges, keep all attention values
            else:
                continue

            # Put the new attention values back into the new alpha tensor
            new_alpha[edge_indices] = node_alpha

        return new_alpha
    def message(self, edge_index_i: Tensor, x_i: Tensor, x_j: Tensor,x:Tensor,edge_index:Tensor,attention:Tensor,
                size_i: Optional[int] ,num_nodes) -> Tensor:
        alpha = self.get_attention(edge_index_i, x_i, x_j,x=x,edge_index=edge_index,attention=attention, num_nodes=size_i)
        if self.mod == "additive":
            ones = torch.ones_like(alpha)
            h = x_j * ones.view(-1, self.heads, 1)
            #h = torch.mul(self.w, h) # 对原始特征进行一次变换，但是w=1，初始是不变
            if self.use_DropKey is False:
                alpha = F.dropout(alpha, p=self.dropout, training=self.training)
            return x_j * alpha.view(-1, self.heads, 1) + (1+self.eps) * h
            
        elif self.mod == "scaled":
            ones = alpha.new_ones(edge_index_i.size())
            degree = scatter_add(ones, edge_index_i, dim_size=num_nodes)[edge_index_i].unsqueeze(-1)
            degree = torch.matmul(degree, self.l1) + self.b1
            degree = self.activation(degree)
            degree = torch.matmul(degree, self.l2) + self.b2
            degree = degree.unsqueeze(-2)
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)
            
            return torch.mul(x_j * alpha.view(-1, self.heads, 1), degree)
            
        elif self.mod == "f-additive":
            alpha = torch.where(alpha > 0, alpha + 1, alpha)
            
        elif self.mod == "f-scaled":
            ones = alpha.new_ones(edge_index_i.size())
            degree = scatter_add(ones, edge_index_i, dim_size=num_nodes)[edge_index_i].unsqueeze(-1)
            alpha = alpha * degree
            
        else:
            alpha = alpha  # origin
        if self.use_DropKey is False:
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.view(-1, self.heads, 1)

    # def update(self, inputs: Tensor) -> Tensor:
    #     r"""Updates node embeddings in analogy to
    #     :math:`\gamma_{\mathbf{\Theta}}` for each node
    #     :math:`i \in \mathcal{V}`.
    #     Takes in the output of aggregation as first argument and any argument
    #     which was initially passed to :meth:`propagate`.
    #     """
    #     return self.lin2(inputs)
    # def update(self, aggr_out, x):
    #     if self.residual:
    #         return (self.beta) * x + aggr_out
    #     else:
    #         return aggr_out

    def negative_sampling(self, edge_index: Tensor, num_nodes: int,
                          batch: OptTensor = None) -> Tensor:

        num_neg_samples = int(self.neg_sample_ratio * self.edge_sample_ratio *
                              edge_index.size(1))

        if not self.is_undirected and not is_undirected(
                edge_index, num_nodes=num_nodes):
            edge_index = to_undirected(edge_index, num_nodes=num_nodes)

        if batch is None:
            neg_edge_index = negative_sampling(edge_index, num_nodes,
                                               num_neg_samples=num_neg_samples)
        else:
            neg_edge_index = batched_negative_sampling(
                edge_index, batch, num_neg_samples=num_neg_samples)

        return neg_edge_index

    def positive_sampling(self, edge_index: Tensor) -> Tensor:
        pos_edge_index, _ = dropout_adj(edge_index,
                                        p=1. - self.edge_sample_ratio,
                                        training=self.training)
        return pos_edge_index
    def sibling_sampling(self, edge_index: Tensor) -> Tensor:
        neu_edge_index, _ = dropout_adj(edge_index,
                                        p=1. - self.sibling_sample_ratio,
                                        training=self.training)
        return neu_edge_index
    def _soft_max_att(self,adj,attention):
        attention=torch.where(adj>0,attention,torch.ones_like(attention)*-9e15)
        return F.softmax(attention,dim=-1)
    def get_attention(self, edge_index_i: Tensor, x_i: Tensor, x_j: Tensor,x:Tensor,edge_index:Tensor,attention:Tensor,
                      num_nodes: Optional[int],
                      return_logits: bool = False) -> Tensor:
        
        
        if self.attention_type == 'MX':
            #print(edge_index_i.shape,x.shape,x_i.shape,x_j.shape,edge_index.shape)
            logits = (x_i * x_j).sum(dim=-1) 
            if return_logits:
                return logits
            x_i = F.leaky_relu(x_i, self.negative_slope)
            x_j = F.leaky_relu(x_j, self.negative_slope)
            alpha = (x_j * self.att_l).sum(-1) + (x_i * self.att_l).sum(-1)#(x_j * self.att_l).sum(-1) + (x_i * self.att_r).sum(-1)
            #print("alpha",alpha.shape,attention.shape)
            alpha = alpha * logits.sigmoid()#alpha * self.rate + logits.sigmoid()#
            if self.use_DropKey == True:
                m_r = torch.ones_like(alpha) * self.dropout
                if self.training:
                    alpha = alpha + torch.bernoulli(m_r) * -1e12

        elif self.attention_type == 'SD':
            alpha = (x_i * x_j).sum(dim=-1) / math.sqrt(self.out_channels)
            if return_logits:
                return alpha
        elif self.attention_type == 'ADD':
            logits = (x_i * x_j).sum(dim=-1) / math.sqrt(self.out_channels)
            if return_logits:
                return logits
            alpha = (x_j * self.att_l).sum(-1) + (x_i * self.att_r).sum(-1)#(x_j * self.att_l).sum(-1) + (x_i * self.att_r).sum(-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            if self.use_DropKey == True:
                m_r = torch.ones_like(alpha) * self.dropout
                if self.training:
                    alpha = alpha + torch.bernoulli(m_r) * -1e12

        #alpha = F.leaky_relu(alpha, self.negative_slope)
        #alpha = self.top_k_attention(alpha,edge_index_i,num_nodes,20)
        # if self.use_topk_softmax:
        #     b = softmax(alpha, edge_index_i, num_nodes=num_nodes)
        #     c=torch.nonzero(b < 0.06)
        #     alpha[c[:,0],c[:,1]] = float('-inf')
        #b[c[:,0],c[:,1]] = float('-inf')
        # for i in range(b.shape[1]):
        #     edge_i = set(np.array(range(b.shape[0]))) - set(np.array(c[c[:, 1] == i][:,0].cpu()))
        #     alpha[:,i] = softmax(alpha[:,i],torch.tensor(edge_i,device=alpha.device), num_nodes = num_nodes)
        #filtered_indices = torch.where(b < 0.18)[0]
        #alpha[alpha < 0.18] = float('-inf')
        #alpha = softmax(alpha, edge_index_i, num_nodes=num_nodes)
        alpha = topk_softmax(alpha, edge_index_i, self.aggr_k, num_nodes=num_nodes)#topk_softmax(alpha, edge_index_i, self.aggr_k, num_nodes=num_nodes)
        return alpha

    def get_attention_loss(self) -> Tensor:
        r"""Compute the self-supervised graph attention loss."""
        if not self.training:
            return torch.tensor([0], device=self.att_l.device)#self.lin.weight.device)

        return F.binary_cross_entropy_with_logits(
            self.att_x.mean(dim=-1),
            self.att_y,
        )

    def get_attention_sibling_loss(self) -> Tensor:
        r"""Compute the self-supervised graph attention loss."""
        if not self.training:
            return torch.tensor([0], device=self.att_l.device)#self.lin.weight.device)

        return F.binary_cross_entropy_with_logits(
            self.att_xs.mean(dim=-1),
            self.att_ys,
        )

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads}, '
                f'type={self.attention_type})')
