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
from torch.nn import Linear, ReLU
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
from torch_geometric.data import Data

model_name = "final_model"

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
epoch = 300





def save_model(m_model, PATH):
    torch.save(m_model.state_dict(), PATH)


def load_model(PATH):
    m_model = Net(dataset.num_node_features, 30, 2).to(device)
    m_model.load_state_dict(torch.load(PATH))
    return m_model


class EdgeConv(MessagePassing):
    def __init__(self, node_channels, edge_channels, out_channels):
        super(EdgeConv, self).__init__() #  "Max" aggregation.
        self.node_channels = node_channels
        self.edge_channels = edge_channels
        self.out_channels = out_channels
        self.lin_n = Linear(node_channels, out_channels, bias=True)
        self.lin_e = Linear(edge_channels, edge_channels , bias=True)
        self.lin_f = Linear(edge_channels , out_channels, bias=True)
        self.relu = ReLU()
        self.relu2 = ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_n.reset_parameters()
        self.lin_e.reset_parameters()
        self.lin_f.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        lisa_print(x, "x")
        lisa_print(edge_attr, "edge_attr")
        x = self.lin_n(x)
        e = self.lin_e(edge_attr)
        x = self.propagate(x = x, edge_index  = edge_index)
        e = self.relu(e)
        lisa_print(e, "e")
        out = self.lin_f(e )
        # out = self.relu2(out)
        lisa_print(out)
        return out
    def aggregate(self, inputs, x,edge_index,  x_i: Tensor, x_j: Tensor) -> Tensor:

        return  torch.abs(x_i - x_j)
    


class Net(torch.nn.Module):
    def __init__(self, node_channels, edge_channels, hidden_channels, num_layers):
        super(Net, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
      
        self.convs.append(EdgeConv(node_channels, edge_channels, 1))

    def forward(self, x, adjs, edge_attr):
        # TODO Here is just a template
        # print("adjs", adjs)
        for i, conv in enumerate(self.convs):
            x = x.float()
            # print("x", x)
            # print("adjs", adjs)
            edge_attr = edge_attr.float()
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
        # loss = F.multilabel_soft_margin_loss(out, data.y)
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

            # loss = F.multilabel_soft_margin_loss(out, data.y)
        except Exception as error:
            raise error
        total_loss += loss.item()
        pred = torch.round(out)
        pred = pred.long().float().flatten()
        y = data.y.flatten()
        n_val_nodes += torch.numel(data.y)
        # print("zz1", pred, y)
        correct += int(pred.eq(y).sum().item())
    acc = correct / n_val_nodes
    hist.add_vl(total_loss / len(val_dataset))
    is_best = hist.add_valid_acc(acc)
    # Save the model if is best
    if is_best:
        if not os.path.exists("checkpoint"):
            os.mkdir("checkpoint")
        file_path = os.path.join("checkpoint", "bestacc.pt")
        save_model(model, file_path)
        save_id += 1
        print(f'Save model at Loss: {total_loss / len(val_dataset):.4f}, Accuracy: {acc:.4f}')
    return total_loss / len(val_dataset), acc


def test(model, test_dataset, device):  # For test, the input data is WHOLE TEST DATASET.
    model.eval()
    correct, n_test_nodes = 0, 0
    for data in test_dataset:
        data = data.to(device)
        pred = model(data.x, data.edge_index, data.edge_attr)
        pred = torch.round(pred)
        pred = pred.long().float().flatten()
        y = data.y.flatten()
        print("diff ", pred, y)
        n_test_nodes += torch.numel(data.y)
        correct += int(pred.eq(y).sum().item())
        correct += int(pred.eq(y+1).sum().item())
        correct += int(pred.eq(y-1).sum().item())
    acc = correct / n_test_nodes

    #     print('No operation accuracy (difference between feature and label): {:.4f}'.format(nop_acc))

    print('Accuracy: {:.4f}'.format(acc))

    return '{:.4f}'.format(acc)


def label2_inference(data: Data,infer_model_name = "final_model"):
    device = torch.device('cpu')
    final_model_path  = pathlib.Path().absolute()
    final_model_path = os.path.join(final_model_path, "Label2/checkpoint/"+infer_model_name+".pt")
    # print(infer_model_name)
    m_model = Net(2, edge_channels = 8, hidden_channels= 1, num_layers = 1).to(device)
    m_model.load_state_dict(torch.load(final_model_path))
    pred = m_model(data.x, data.edge_index, data.edge_attr)
    pred = torch.round(pred)
    return pred


if __name__ == "__main__":

    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        data_path = os.path.join(path.parent.parent, 'data/training_dataset')
        data_path = os.path.join(data_path,model_name )

    processed = os.path.join(data_path, 'processed')
    if os.path.exists(processed):
        shutil.rmtree(processed)
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

    print("num_node_features", dataset.num_node_features)
    print("num_edge_features", dataset.num_edge_features)


    
    ##################### Main function ####################
    device = torch.device('cpu')
    model = Net(dataset.num_node_features,dataset.num_edge_features, 1, 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-3)
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
    acc = test(model, test_dataset, device)
    file_path = os.path.join("checkpoint", model_name+".pt")
    save_model(model, file_path)
    print(f'Save the final model!')

    acc_file = open('../accuracy_log.txt', 'a+')
    acc_file.write(model_name + " 2 "+ str(acc)+"\n")
    acc_file.close()

    # !!!! Remove the preprocessed folder AUTOMATICALLY!!!
    processed = os.path.join(data_path, 'processed')
    shutil.rmtree(processed)
