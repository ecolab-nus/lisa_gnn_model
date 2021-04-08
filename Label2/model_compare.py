import sys

sys.path.append('../../dfg_generator')
sys.path.append('../../dfg_generator/dfg')
sys.path.append('../../dfg_generator/graph_generation')

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
from torch_geometric.data import NeighborSampler as RawNeighborSampler, Batch, DataLoader
from torch_geometric.utils import degree
from torch_geometric.utils import add_self_loops
import matplotlib.pyplot as plt
import numpy as np
import shutil

path = pathlib.Path().absolute()
data_path = os.path.join(path.parent.parent, 'data')

####################### Parameter Setting ###################################
val_freq = 50  # Do validation for every [val_freq] epochs
# 'label_indicator' indicates which features to use as train label
# 0: schedule order,
# 1: communication
# 2: start node distance
# 3: neighbour distance
label_indicator = 2
batch_size = 10
epoch = 100

####################### Dataset Loading ######################################
dataset = dfg_dataset(data_path, label_indicator)
dataset = dataset.shuffle()

####################### Dataset Loading ######################################
dataset = dfg_dataset(data_path, label_indicator)
dataset = dataset.shuffle()
data_loader = DataLoader(dataset, batch_size=batch_size)


def save_model(m_model, PATH):
    torch.save(m_model.state_dict(), PATH)

def load_model(PATH):
    m_model = Net(dataset.num_node_features, 30, 2).to(device)
    m_model.load_state_dict(torch.load(PATH))
    return m_model

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
                 bias: bool = True, **kwargs):  # yapf: disable
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
        # self.lin_f = Linear(1, 1, bias=True)
        self.lin_f = Linear(out_channels, out_channels, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        # self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        add_self_loops(edge_index, num_nodes=len(x[0]))
        x_r = x[1]
        # print("x_r", x_r.size())
        deg = degree(edge_index[1], num_nodes=len(x[0]))
        deg = deg.clamp_(1).view(-1, 1)
        # print("deg", deg.size())
        out = self.lin_l(deg)

        # print("out", out.size())

        if x_r is not None:
            out += x_r

        if self.normalize:
            out = F.normalize(out, p=2., )
        # print("out", out.size())
        lin_f = self.lin_f(out)
        return out

    def message(self, x_i, x_j, norm):
        return norm.view(-1, 1) * x_j

    def aggregate(self, inputs: Tensor, x: Union[Tensor, OptPairTensor], index: Tensor) -> Tensor:
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
            # out_channels = 1 if i == num_layers - 1 else 1
            out_channels = 3
            self.convs.append(LISAConv(1, out_channels, normalize=False))

    def forward(self, x, adjs, input):
        # TODO Here is just a template
        # print("adjs", adjs)
        for i, conv in enumerate(self.convs):
            x = conv(x, adjs)
            if i != self.num_layers - 1:
                # x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
            # print("x",i, x)
        x = torch.flatten(x)
        out = torch.sum(x).reshape((1))
        return out

def test(model, test_dataset, device): # For test, the input data is WHOLE TEST DATASET.
    model.eval()
    correct, n_test_nodes = 0, 0
    for data in test_dataset:
        data = data.to(device)
        y = torch.reshape(data.y, (-1, 3))
        for i in range(len(y)):
            out = model(data.x, data.edge_index, y[i][:2])
            pred = torch.round(out).long().float()
            y = y[i][2:]
            n_test_nodes += 1
            correct += int(pred.eq(y).sum().item())
    acc = correct / n_test_nodes
    print('Accuracy: {:.4f}'.format(acc))

##################### Main function ####################
device = torch.device('cpu')
####################### Model Testing #############################
test_model_path = os.path.join("checkpoint", "m_1_0.0.pt")  #### MODIFY THIS PLEASE
final_model_path = os.path.join("checkpoint", "final_model.pt")
model0 = load_model(test_model_path)
model1 = load_model(final_model_path)
print("Accuracy for checkpoint model:")
test(model0, dataset, device)
print("Accuracy for final model:")
test(model1, dataset, device)

# !!!! Remove the preprocessed folder AUTOMATICALLY!!!
processed = os.path.join(data_path, 'processed')
shutil.rmtree(processed)