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
label_indicator = 3
batch_size = 10
epoch = 100

####################### Dataset Loading ######################################
dataset = dfg_dataset(data_path, label_indicator)
dataset = dataset.shuffle()

####################### Data loader for minibatch #############################
# test:validation:train = 1:1:8
test_dataset = dataset[:len(dataset) // 10]
val_dataset = dataset[len(dataset) // 10:len(dataset) // 5]
train_dataset = dataset[len(dataset) // 5:]

test_loader = DataLoader(test_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size)


class history():
    def __init__(self):
        self.train_loss = []
        self.valid_loss = []
        self.valid_acc = []
        self.best_val_acc = 0

    def add_tl(self, loss):  # add train loss (average for each graph, so that the batch size doesn't matter)
        self.train_loss.append(loss)
        # Don't save the model during training, otherwise too many "saving" at the beginning

    def add_vl(self, loss):  # add validation loss
        self.valid_loss.append(loss)

    def add_valid_acc(self, acc):
        self.valid_acc.append(acc)
        if acc >= self.best_val_acc:
            self.best_val_acc = acc
            return True
        else:
            return False

    def plot_hist(self):
        fig, ax = plt.subplots()  # Create a figure containing a single axes.
        ax.plot(range(len(self.train_loss)), self.train_loss, label="train")
        ax.plot([val_freq * x for x in range(len(self.valid_loss))], self.valid_loss, label="validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Avg Loss per graph")
        ax.set_title("Loss History")
        ax.legend()
        print("Save to loss_history.jpg")
        plt.savefig("loss_history.jpg")

        fig2, ax2 = plt.subplots()  # Create a figure containing a single axes.
        ax2.plot([val_freq * x for x in range(len(self.valid_acc))], self.valid_acc, label="validation")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Accuracy History")
        ax2.legend()
        print("Save to acc_history.jpg")
        plt.savefig("acc_history.jpg")


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
        out = torch.tensor([torch.sum(out), torch.sum(out)-1])
        return out


def train(model, data, device, optimizer):  # For training, the input data is a BATCH DATA iterated from DATALOADER.
    model.train()
    optimizer.zero_grad()
    data = data.to(device)
    y = torch.reshape(data.y, (-1, 4))  # TODO data.y should be (1, 4)? not (Any, 4)
    for i in range(len(y)):
        out = model(data.x, data.edge_index, y[i][:2])
        try:
            loss = F.mse_loss(out, y[i][2:], reduction='mean')
        except Exception as error:
            raise error
        loss.backward()
        optimizer.step()
    return loss / data.num_graphs


def validation(model, val_dataset, device):  # For validation, the input data is WHOLE VALIDATION DATASET.
    model.eval()
    global save_id
    total_loss = 0
    correct, n_val_nodes = 0, 0
    for data in val_dataset:
        # print(data.num_graphs)
        data = data.to(device)
        y = torch.reshape(data.y, (-1, 4))  # TODO data.y should be (1, 4)? not (Any, 4)
        for i in range(len(y)):
            out = model(data.x, data.edge_index, y[i][:2])
            try:
                loss = F.mse_loss(out, y[i][2:], reduction='mean')
            except Exception as error:
                raise error
            total_loss += loss.item()
            pred = torch.round(out).long().float()
            y = y[i][2:]
            n_val_nodes += 1
            correct += int(pred.eq(y).sum().item())
    acc = correct / n_val_nodes
    hist.add_vl(total_loss / len(val_dataset))
    is_best = hist.add_valid_acc(acc)
    # Save the model if is best
    if is_best:
        if not os.path.exists("checkpoint"):
            os.mkdir("checkpoint")
        file_path = os.path.join("checkpoint", "m_" + str(save_id) + "_" + str(acc) + ".pt")
        save_model(model, file_path)
        save_id += 1
        print(f'Save model at Loss: {total_loss / len(val_dataset):.4f}, Accuracy: {acc:.4f}')
    return total_loss / len(val_dataset), acc


def test(model, test_dataset, device):  # For test, the input data is WHOLE TEST DATASET.
    model.eval()
    correct, n_test_nodes = 0, 0
    for data in test_dataset:
        data = data.to(device)
        y = torch.reshape(data.y, (-1, 4))  # TODO data.y should be (1, 4)? not (Any, 4)
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
model = Net(dataset.num_node_features, 30, 2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
save_id = 0
####################### Model Testing #############################
hist = history()
for i in range(epoch):
    if (i % val_freq):
        loss = 0
        # For whole train set do training
        for data in train_loader:
            # For each batch do training
            loss += train(model, data, device, optimizer)
        print("Epoch %d, Loss %.6f" % (i, loss))
        hist.add_tl(loss / len(train_loader))
    else:
        # For whole validation set do training
        loss, acc = validation(model, val_dataset, device)
        print("#Val# Epoch %d, Loss %.6f, Acc %.6f" % (i, loss, acc))

hist.plot_hist()
test(model, test_dataset, device)
file_path = os.path.join("checkpoint", "final_model.pt")
save_model(model, file_path)
print(f'Save the final model!')

# !!!! Remove the preprocessed folder AUTOMATICALLY!!!
processed = os.path.join(data_path, 'processed')
shutil.rmtree(processed)
