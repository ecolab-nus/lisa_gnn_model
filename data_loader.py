import os
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import numpy as np
from data_generator import generator


def load_data(graph_dir, label_dir):
    """
    Function:
        Load data set into a list of torch_geometric.data.Data objects
    Params:
        graph_dir: str, directory to the graph (edge) information,
        label_dir: str, directory to the label information,
    Return:
        list of torch_geometric.data.Data objects
    """
    dataset = []
    graph_files = os.listdir(graph_dir)
    for file in graph_files:
        if not file.endswith(".txt"):
            continue

        # Get edge data
        edge = []
        f_graph = open(os.path.join(graph_dir, file), 'r')
        for line in f_graph:
            a, b = line.strip().split('\t')
            edge.append([int(a), int(b)])
        edge = torch.tensor(edge, dtype=torch.long)
        edge_index = edge.t().contiguous()
        f_graph.close()

        # Get label data
        x = []
        y = []
        f_label = open(os.path.join(label_dir, file), 'r')
        for line in f_label:
            a, b = line.strip().split('\t')
            x.append([int(a)])
            y.append(int(b))
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)
        f_label.close()

        data = Data(x=x, edge_index=edge_index, y=y)
        dataset.append(data)
    return dataset

class dfg_dataset(InMemoryDataset):

    def __init__(self, root, transform=None, pre_transform=None):
        super(dfg_dataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        graph_files = [os.path.join("graph", str(i)+'.txt') for i in range(1000)]
        return graph_files

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        generator(1000, self.raw_dir)

    def process(self):
        # Read data into huge `Data` list.
        data_list = load_data(os.path.join(self.raw_dir, "graph"), os.path.join(self.raw_dir, "label"))
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

