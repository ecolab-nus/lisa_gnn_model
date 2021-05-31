import sys
sys.path.append('../')
sys.path.append('../../dfg_generator')
sys.path.append('../../dfg_generator/dfg')
sys.path.append('../../dfg_generator/graph_generation')

from data_loader import dfg_dataset
from history import history

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
from torch_geometric.data import Data
path = pathlib.Path().absolute()
data_path = os.path.join(path.parent.parent, 'data')

####################### Parameter Setting ###################################
val_freq = 20  # Do validation for every [val_freq] epochs
# 'label_indicator' indicates which features to use as train label
# 0: schedule order,
# 1: communication
# 2: start node distance
# 3: neighbour distance
label_indicator = 0
batch_size = 10
epoch = 1000



model_name = "final_model"


def save_model(m_model, PATH):
    torch.save(m_model.state_dict(), PATH)

def load_model(PATH):
    m_model = Net(dataset.num_node_features, 30, 2).to(device)
    m_model.load_state_dict(torch.load(PATH))
    return m_model

class LISASchedConv(MessagePassing):

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 bias: bool = True, **kwargs):  # yapf: disable
        kwargs.setdefault('aggr', None)
        super(LISASchedConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        # print("out_channels",out_channels)

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

       
        # self.lin_r = Linear(in_channels[1], out_channels, bias=True)
        self.lin_a = Linear(1, 1, bias=False)
        self.lin_1 = Linear(in_channels[0] - 1, 1, bias=True)
        self.side_feature = in_channels[0] - 1
       
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_a.reset_parameters()
        self.lin_1.reset_parameters()
        

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        # print("!!!!!!!!!!!!!!!!")
        # print("side_feature", self.side_feature)
        # print("edge", edge_index)
        
        if isinstance(x, Tensor):
            # print("x", x.size())
            x: OptPairTensor = (x[:, 0], x[:, 1:])
        

        temp_edge_index =  to_undirected(edge_index, len(x[0]) )
        # print("unedge", temp_edge_index)

        # print("x", x[0].size())
        # print("x[1]", x[1], x[1].size())

        neigbor_info = self.propagate(x = (x[1], x[1]), edge_index = temp_edge_index,  deg_trans = x[1] )
        # x [1] = neigbor_info
         
        trans_neig  = self.lin_1(neigbor_info)
        ori  =  x[0].view(-1, 1)
        result = self.lin_a(ori) + trans_neig

        x: OptPairTensor = (result, trans_neig)
        
        return x

    def aggregate(self, inputs, x: Union[Tensor, OptPairTensor],edge_index,  deg_trans_i: Tensor,) -> Tensor:
        # print("deg_trans_i",deg_trans_i,  deg_trans_i.size())
        # print("edge_index[1]", edge_index[1], edge_index[1].size())
        # index  =edge_index[1].resize(1, len(edge_index[1]))
        index  = torch.zeros(self.side_feature, len(edge_index[0]))
        for i in range(0, self.side_feature):
            index [i] =  edge_index[0]
       
        deg_trans_i = torch.transpose(deg_trans_i, 0 , 1)
        index = index.long()
        # print("index", index, index.size())
        neigbor_degree_sum = scatter(deg_trans_i, index = index,  dim =1,  reduce="sum") 
        # neigbor_degree_sum2 = scatter(deg_trans_j[:,1], index = edge_index[1],   reduce="sum") 
        # neigbor_degree_sum3 = scatter(deg_trans_j[:,2], index = edge_index[1],   reduce="sum") 
        
        neigbor_degree_sum = torch.transpose(neigbor_degree_sum, 0, 1)
        # print("neigbor_degree_sum", neigbor_degree_sum, neigbor_degree_sum.size())
        return neigbor_degree_sum

        
    def old_aggregate(self, inputs, x: Union[Tensor, OptPairTensor], und_edge_index: Tensor, deg_trans_j: Tensor,) -> Tensor:
        

        # print("dim size", len(x[0]))
        # print("und_edge_index[1]", und_edge_index[1].size())
        # print("deg_trans_j", deg_trans_j.size())
        deg_trans_j  =deg_trans_j.resize(1, len(deg_trans_j))
      
        # print("index", index.size())
        neigbor_degree_sum = scatter(deg_trans_j, index = und_edge_index[1],  dim_size = len(x[0]),  reduce="sum") 
        # print("parent_val", parent_val.size(), parent_val)
        # print("child_val", child_val.size(), child_val)
        # print("(neigbor_degree_sum", neigbor_degree_sum.size(), neigbor_degree_sum)
        neigbor_degree_sum = neigbor_degree_sum.resize(len(x[0]), 1)
        neigbor_degree_sum =  self.lin_nn(neigbor_degree_sum)

        deg = degree(und_edge_index[0], num_nodes=len(x[0]))
        deg = deg.clamp_(1).view(-1, 1)
        deg = self.lin_n(deg)

        neigbor_degree_sum = neigbor_degree_sum.add(deg)
        
        # child_val = child_val.resize(len(x[0]), 1)
        final =  self.lin_d(x[0]).add(neigbor_degree_sum)
        result: OptPairTensor = (final, neigbor_degree_sum)
        # print("final", final.size(), final)
        return result

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)



class Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers):
        super(Net, self).__init__()
        self.num_layers = num_layers
        self.convs = nn.ModuleList()
       
        self.convs.append(LISASchedConv(in_channels, 2, normalize=False))
        self.convs.append(LISASchedConv(2, 2, normalize=False))

    def forward(self, x, adjs):
        # print("adjs", adjs)
        # print("x", x)
        x = x.long().float()
        for i, conv in enumerate(self.convs):
            x = conv(x, adjs)
            # if i != self.num_layers - 1:
            #     # x = x.relu()
            #     x = F.dropout(x, p=0.5, training=self.training)
            # print("x",i, x)
        x = torch.flatten(x[0])
        return x


def train(model, data, device, optimizer):  # For training, the input data is a BATCH DATA iterated from DATALOADER.
    model.train()
    optimizer.zero_grad()
    data = data.to(device)
    out = model(data.x, data.edge_index)
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
        out = model(data.x, data.edge_index)
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
        pred = model(data.x, data.edge_index)
        pred = torch.round(pred)
        pred = pred.long().float().flatten()
        y = data.y.flatten()
        print("diff ", pred, y)
        n_test_nodes += torch.numel(data.y)
        correct += int(pred.eq(y).sum().item())
    acc = correct / n_test_nodes

    #     print('No operation accuracy (difference between feature and label): {:.4f}'.format(nop_acc))

    print('Accuracy: {:.4f}'.format(acc))

def label0_inference(data: Data):
    device = torch.device('cpu')
    final_model_path  = pathlib.Path().absolute()
    final_model_path = os.path.join(final_model_path, "Label0/checkpoint/"+model_name+".pt")
    m_model = Net(5, 30, 2).to(device)
    m_model.load_state_dict(torch.load(final_model_path))
    pred = m_model(data.x, data.edge_index)
    pred = torch.round(pred)
    return pred


if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
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


    ##################### Main function ####################
    device = torch.device('cpu')
    # model = SAGENet(dataset.num_node_features, hidden_channels=64, out_channels= 1, num_layers=2)

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
    file_path = os.path.join("checkpoint", model_name+".pt")
    save_model(model, file_path)
    print(f'Save the final model!')

    use_check = False
    if use_check:
        best_acc = os.path.join("checkpoint", "bestacc.pt")
        model0 = load_model(best_acc)
        print("Accuracy for checkpoint model:")
        test(model0, test_dataset, device)
    # !!!! Remove the preprocessed folder AUTOMATICALLY!!!
    processed = os.path.join(data_path, 'processed')
    shutil.rmtree(processed)
