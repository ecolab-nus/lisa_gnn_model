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

path = pathlib.Path().absolute()
data_path = os.path.join(path.parent, 'data')
dataset = dfg_dataset(data_path, 100, 0)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

device = torch.device('cpu')
model = Net().to(device)
dataset_split_pt = int(0.9*dataset.num_data)  # decide the split point for train\test set
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    for i in range(dataset_split_pt):
        data = dataset[i].to(device)
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()

model.eval()
correct, nop_correct, n_test_nodes = 0, 0, 0
for i in range(dataset.num_data-dataset_split_pt):
    data = dataset[dataset_split_pt+i].to(device)
    _, pred = model(data).max(dim=1)
    n_test_nodes += len(data.y)
    correct += int(pred.eq(data.y).sum().item())
    nop_correct += data.x.T[0].eq(data.y).sum().item()

nop_acc = nop_correct / n_test_nodes
acc = correct / n_test_nodes
print('No operation accurarcy (difference between feature and label): {:.4f}'.format(nop_acc))
print('Accuracy: {:.4f}'.format(acc))
