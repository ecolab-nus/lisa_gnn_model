import sys
sys.path.append('../')
sys.path.append('../../dfg_generator')
sys.path.append('../../dfg_generator/dfg')
sys.path.append('../../dfg_generator/graph_generation')

from data_loader import dfg_dataset
from history import history
from history import lisa_print

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
from torch_geometric.utils import to_undirected
from torch_scatter import gather_csr, scatter, segment_csr
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
val_freq = 20  # Do validation for every [val_freq] epochs
# 'label_indicator' indicates which features to use as train label
# 0: schedule order,
# 1: communication
# 2: start node distance
# 3: neighbour distance
label_indicator = 1
batch_size = 10
epoch = 1000

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



def save_model(m_model, PATH):
    torch.save(m_model.state_dict(), PATH)

def load_model(PATH):
    m_model = Net(dataset.num_node_features, 30, 2).to(device)
    m_model.load_state_dict(torch.load(PATH))
    return m_model
    
class LISACommConv(MessagePassing):

    def __init__(self, node_channels, edge_channels, out_channels):
        super(LISACommConv, self).__init__()

        self.in_channels = edge_channels
        self.out_channels = out_channels
        self.lin_a = Linear(edge_channels, 1, bias=True)
        self.lin_d = Linear(1, 1, bias=True)


        self.reset_parameters()

    def reset_parameters(self):
        self.lin_a.reset_parameters()
        self.lin_d.reset_parameters()

    def forward(self, x, edge_index: Adj, edge_attr) -> Tensor:
        
        edge_value = self.lin_a(edge_attr)
        lisa_print (edge_value, "edge_value")
        neigbor_value = self.propagate(x = x, edge_index = edge_index, edge_value =edge_value)
        out =  self.lin_d(neigbor_value)
        return out


    def aggregate(self, inputs, x, edge_index,  edge_value: Tensor) -> Tensor:
        lisa_print(edge_value, "edge_value")
        edge_value = edge_value.view(-1)
        neigbor_value = scatter(edge_value, index = edge_index[1] ,  reduce="sum")
        neigbor_value = neigbor_value.view(-1, 1)
        return neigbor_value



class LISACommConv1(MessagePassing):

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 bias: bool = True, **kwargs):  # yapf: disable
        kwargs.setdefault('aggr', None)
        super(LISACommConv1, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.lin_a = Linear(1, 1, bias=True)
        self.lin_d = Linear(1, 1, bias=True)
        self.lin_asap = Linear(1, 1, bias=True)

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_a.reset_parameters()
        self.lin_d.reset_parameters()
        self.lin_asap.reset_parameters()

    def forward(self, x, edge_index: Adj, size: Size = None) -> Tensor:
        ancestor = x[:, 0] 
        ancestor = torch.reshape(ancestor,(len(ancestor), 1))
        descendant = x[:, 1] 
        descendant = torch.reshape(descendant,(len(descendant), 1))
        asap = x[:, 2] 
        asap = asap.view(-1, 1)
        lisa_print(ancestor, "ancestor")
        lisa_print(descendant)
        out = self.propagate(x = x, ancestor =  ancestor, descendant = descendant, edge_index = edge_index, asap = asap)
        return out


    def aggregate(self, inputs, x, edge_index,  ancestor_j: Tensor, descendant_i: Tensor, asap_i , asap_j) -> Tensor:
        lisa_print(ancestor_j, "ancestor_j")
        lisa_print(edge_index, "edge_index")
        ancestor_j = ancestor_j.view(len(ancestor_j))
        descendant_i =  descendant_i.view(len(descendant_i))
        parent_sum = scatter(ancestor_j, index = edge_index[1],  dim_size = len(x) , reduce="sum") 
        child_sum = scatter(descendant_i, index = edge_index[0],   dim_size = len(x) ,  reduce="sum")
        parent_sum = parent_sum.view(-1, 1)
        child_sum = child_sum.view(-1, 1)
        lisa_print(parent_sum, "parent_sum")

        asap_i = asap_i.view(-1)
        asap_j = asap_j.view(-1)
        asap_diff = torch.abs(asap_i - asap_j)
        lisa_print(asap_diff, "asap_diff")
        # asap_diff =  asap_diff.view(-1, 1)
        asap_sum1 = scatter(asap_diff, index = edge_index[0],   dim_size = len(x) ,  reduce="sum")
        asap_sum2 = scatter(asap_diff, index = edge_index[1],   dim_size = len(x) ,  reduce="sum")
        asap_sum = asap_sum1 +  asap_sum2
        asap_sum = asap_sum.view(-1, 1)

        # print("parent_sum", parent_sum.size())
        # print("child_sum", child_sum.size())

        out =  self.lin_asap(asap_sum)
        return out



    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class LISACommConv2(MessagePassing):

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 bias: bool = True, **kwargs):  # yapf: disable
        kwargs.setdefault('aggr', None)
        super(LISACommConv2, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        # print("out_channels",out_channels)
        self.lin_l = Linear(1, 1, bias=True)
        self.lin_r = Linear(1, 1, bias=True)
        self.lin_d = Linear(1, 1, bias=True)

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        
        # self.lin_r = Linear(in_channels[1], out_channels, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x, edge_index: Adj,
                size: Size = None) -> Tensor:

       
        feature = self.lin_l(x)
        und_edge_index =  to_undirected(edge_index, len(x) )
        neigbor_info = self.propagate(x = x, edge_index = und_edge_index )
        neigbor_info = self.lin_r(neigbor_info)

        deg = degree(und_edge_index[0], x.size(0), dtype=x.dtype)
        deg = deg.view(-1, 1)

        final_info = self.lin_d(deg) + feature + neigbor_info
        
        return final_info

    def aggregate(self, inputs, x,edge_index,  x_i: Tensor) -> Tensor:
        data = x_i.view(-1)
        lisa_print( data)       
        neigbor_degree_sum = scatter(data, index = edge_index [0],   reduce="sum") 
        neigbor_degree_sum = neigbor_degree_sum.view(-1,1)
        lisa_print(neigbor_degree_sum)
        
        return neigbor_degree_sum

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class Net(torch.nn.Module):
    def __init__(self, node_channels,edge_channels, hidden_channels, num_layers):
        super(Net, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
        
        self.convs.append(LISACommConv(node_channels, edge_channels, 1))

    def forward(self, x, adjs, edge_attr):
        x = x.long().float()
        edge_attr = edge_attr.float()
        for i, conv in enumerate(self.convs):
            x = conv(x, adjs, edge_attr)
            if i != self.num_layers - 1:
                # x = x.relu()
                x = F.dropout(x, p=0.5, training=self.training)
            # print("x",i, x)
        x = torch.flatten(x)
        return x


def train(model, data, device, optimizer):  # For training, the input data is a BATCH DATA iterated from DATALOADER.
    model.train()
    optimizer.zero_grad()
    data = data.to(device)
    out = model(data.x, data.edge_index, data.edge_attr)
    try:
        loss = F.mse_loss(out, data.y, reduction='mean')
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
        out = model(data.x, data.edge_index, data.edge_attr)
        try:
            loss = F.mse_loss(out, data.y, reduction='mean')
        except Exception as error:
            raise error
        total_loss += loss.item()
        pred = torch.round(out)
        pred = pred.long().float().flatten()
        y = data.y.flatten()
        n_val_nodes += torch.numel(data.y)
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
        pred = model(data.x, data.edge_index,  data.edge_attr)
        pred = torch.round(pred)
        pred = pred.long().float().flatten()
        y = data.y.flatten()
        print("diff ", pred, y)
        n_test_nodes += torch.numel(data.y)
        correct += int(pred.eq(y).sum().item())
    acc = correct / n_test_nodes

    #     print('No operation accuracy (difference between feature and label): {:.4f}'.format(nop_acc))

    print('Accuracy: {:.4f}'.format(acc))


##################### Main function ####################
device = torch.device('cpu')
model = Net(dataset.num_node_features,dataset.num_edge_features,  1, 1).to(device)
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