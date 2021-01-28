import sys
sys.path.append('../dfg_generator')
sys.path.append('../dfg_generator/dfg')
sys.path.append('../dfg_generator/graph_generation')

from data_loader import dfg_dataset

import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import pathlib

from typing import Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, Size
from torch import Tensor
from torch.nn import Linear
import torch.nn.functional as F
# from torch_sparse import SparseTensor
from torch_geometric.nn.conv import MessagePassing



import torch.nn as nn
from torch_cluster import random_walk
from sklearn.linear_model import LogisticRegression

import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv
from torch_geometric.datasets import Planetoid
from torch_geometric.data import NeighborSampler as RawNeighborSampler
from torch_geometric.utils import degree
from torch_geometric.utils import add_self_loops
path = pathlib.Path().absolute()
data_path = os.path.join(path.parent, 'data')
dataset = dfg_dataset(data_path, 100, 0)

class LISAConv(MessagePassing):
    """The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{W}_1 \mathbf{x}_i + \mathbf{W_2} \cdot
        \mathrm{mean}_{j \in \mathcal{N(i)}} \mathbf{x}_j

    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized, *i.e.*,
            :math:`\frac{\mathbf{x}^{\prime}_i}
            {\| \mathbf{x}^{\prime}_i \|_2}`.
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 bias: bool = True,  **kwargs):  # yapf: disable
        kwargs.setdefault('aggr', None)
        super(LISAConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        # print("out_channels",out_channels)

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=True)
        # self.lin_r = Linear(in_channels[1], out_channels, bias=True)
        self.lin_f = Linear(1, 1, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        # self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        add_self_loops(edge_index, num_nodes = len(x[0]))
        x_r = x[1]
        # print("x_r", x_r.size())
        deg = degree(edge_index[1],  num_nodes = len(x[0]))
        deg = deg.clamp_(1).view(-1,  1)
        # print("deg", deg.size())
        out = self.lin_l(deg)

        # print("out", out.size())
       
        if x_r is not None:
            out += x_r
        
        if self.normalize:
            out = F.normalize(out, p=2.,)
        # print("out", out.size())
        lin_f = self.lin_f(out)
        return out

    def message(self,x_i, x_j, norm):
        return norm.view(-1, 1) * x_j


    def aggregate(self,  inputs: Tensor, x : Union[Tensor, OptPairTensor], index: Tensor) -> Tensor:
        return inputs


    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super(Net, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            # in_channels = in_channels if i == 0 else 1
            out_channels = 1 if i == num_layers-1 else 1
            self.convs.append(LISAConv(1, out_channels, normalize =  False ))

    def forward(self, x, adjs):
        # print("adjs", adjs)
        for i, conv in enumerate(self.convs):
            x = conv(x, adjs)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
            # print("x",i, x)
        x = torch.flatten(x)
        return x

 

device = torch.device('cpu')
model = Net(dataset.num_node_features, 30, 2).to(device)
dataset_split_pt = int(0.9*dataset.num_data)  # decide the split point for train\test set
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

model.train()


for epoch in range(100):
    optimizer.zero_grad()
    total_loss = 0
    for i in range(dataset_split_pt):
        # print("graph index", i)
        data = dataset[i].to(device)
        # print("data.edge_index", len(data.x), data.edge_index.size())
        out = model(data.x, data.edge_index)
        try:
            loss = F.mse_loss(out, data.y, reduction = 'mean')
        except Exception as error:
            print(i)
            raise error

        total_loss += float(loss)
        loss.backward()
        optimizer.step()
    if epoch % 5 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {total_loss:.4f}')

model.eval()
correct, nop_correct, n_test_nodes = 0, 0, 0
for i in range(dataset.num_data-dataset_split_pt):
    data = dataset[dataset_split_pt+i].to(device)
    pred = model(data.x, data.edge_index)
    pred = torch.round(pred)
    pred = pred.long()
    pred = pred.float()
    n_test_nodes += len(data.y)
    print("pred", pred)
    print("data.y", data.y)
    correct += int(pred.eq(data.y).sum().item())
    nop_correct += data.x.T[0].eq(data.y).sum().item()

nop_acc = nop_correct / n_test_nodes
acc = correct / n_test_nodes
print('No operation accurarcy (difference between feature and label): {:.4f}'.format(nop_acc))
print('Accuracy: {:.4f}'.format(acc))
