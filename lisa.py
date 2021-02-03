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

import matplotlib.pyplot as plt
import numpy as np

path = pathlib.Path().absolute()
data_path = os.path.join(path.parent, 'data')

val_freq = 50  # Do validation for every [val_freq] epochs
num_graphs = 10000  # number of graphs to load into whole dataset
label_indicator = 0  # indicate which column to use as training label
dataset = dfg_dataset(data_path, num_graphs, label_indicator)
portion = [0.7, 0.1, 0.2]  # Split dataset into train\validation\test.
datasets_len = [int(x*num_graphs) for x in portion]

class history():
    def __init__(self):
        self.train_loss = []
        self.valid_loss = []
        self.valid_acc = []
    def add_tl(self, loss): # add train loss (average for each graph, so that the batch size doesn;t matter)
        self.train_loss.append(loss)
    def add_vl(self , loss):  #  add validation loss
        self.valid_loss.append(loss)
    def add_valid_acc(self, acc):
        self.valid_acc.append(acc)
    def plot_hist(self):
        # TODO, plot the history for loss
        fig, ax = plt.subplots()  # Create a figure containing a single axes.
        ax.plot(range(len(self.train_loss)), self.train_loss, label="train")
        ax.plot([val_freq*(x+1) for x in range(len(self.valid_loss))], self.valid_loss, label="validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Avg Loss per graph")
        ax.set_title("Loss History")
        ax.legend()
        plt.show()
    def check_point(self):
        # TODO, save  the best model
        pass

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
                # x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
            # print("x",i, x)
        x = torch.flatten(x)
        return x

 

device = torch.device('cpu')
model = Net(dataset.num_node_features, 30, 2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
model.train()

hist = history()
for epoch in range(100):
    print(epoch)
    if epoch % val_freq == val_freq-1:  # Do validation, turn mode to evaluation
        model.eval()
        total_loss = 0
        correct, n_val_nodes = 0, 0
        for i in range(datasets_len[0], sum(datasets_len[:2])):  # For data belong to training
            data = dataset[i].to(device)
            out = model(data.x, data.edge_index)
            try:
                loss = F.mse_loss(out, data.y, reduction='mean')
            except Exception as error:
                raise error
            total_loss += float(loss)
            pred = torch.round(out)
            pred = pred.long().float()
            n_val_nodes += len(data.y)
            correct += int(pred.eq(data.y).sum().item())
        acc = correct / n_val_nodes
        print(f'#VALIDATION# Epoch: {epoch:03d}, Loss: {total_loss:.4f}, Accuracy: {acc:.4f}')
        hist.add_vl(total_loss/datasets_len[1])
        hist.add_valid_acc(acc)
    else:  # Do training, turn mode to train
        optimizer.zero_grad()
        model.train()
        total_loss = 0
        for i in range(datasets_len[0]):  # For data belong to training
            data = dataset[i].to(device)
            out = model(data.x, data.edge_index)
            try:
                loss = F.mse_loss(out, data.y, reduction='mean')
            except Exception as error:
                raise error
            total_loss += float(loss)
            loss.backward()
            optimizer.step()
        print(f'Epoch: {epoch:03d}, Loss: {total_loss:.4f}')
        hist.add_tl(total_loss/datasets_len[0])

hist.plot_hist()
model.eval()
correct, nop_correct, n_test_nodes = 0, 0, 0
for i in range(sum(datasets_len[:2]), sum(datasets_len[:3])):
    data = dataset[i].to(device)
    pred = model(data.x, data.edge_index)
    pred = torch.round(pred)
    pred = pred.long()
    pred = pred.float()
    n_test_nodes += len(data.y)
    correct += int(pred.eq(data.y).sum().item())
    nop_correct += data.x.T[0].eq(data.y).sum().item()

nop_acc = nop_correct / n_test_nodes
acc = correct / n_test_nodes
print('No operation accurarcy (difference between feature and label): {:.4f}'.format(nop_acc))
print('Accuracy: {:.4f}'.format(acc))
