import os
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import numpy as np
from data_generator import generator
from torch_geometric.utils import contains_self_loops
from torch_geometric.utils import degree

def load_data(graph_dir, label_dir, id_label):
    """
    Function:
        Load data set into a list of torch_geometric.data.Data objects
    Params:
        graph_dir: str, directory to the graph (edge) information,
        label_dir: str, directory to the label information,
        id_label: int, index of label. There may be multiple labels, id_label indicates which label we want to use for GNN.
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
        if contains_self_loops(edge_index):
            print(edge_index)
            assert False
        
        f_graph.close()

        # Get label data
        x = []
        y = []
        f_label = open(os.path.join(label_dir, file), 'r')
        for line in f_label:
            a, b = line.strip().split('#')
            a = a.split('\t')
            b = b.split('\t')
            x.append([int(m_x) for m_x in a])
            y.append(int(b[id_label]))
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.float)
        f_label.close()


        # deg = degree(edge_index)
        # if len(deg) != len(x):
        #     assert (False)
        data = Data(x=x, edge_index=edge_index, y=y)
        dataset.append(data)
    return dataset

class dfg_dataset(InMemoryDataset):

    def __init__(self, root, num_data, id_label, transform=None, pre_transform=None):
        """

        :param root: str, root of dataset
        :param num_data: int, number of data points generated
        :param id_label: index of label to use for GNN
        :param transform:
        :param pre_transform:
        """
        self.num_data = num_data
        self.id_label = id_label
        super(dfg_dataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        graph_files = [os.path.join("graph", str(i)+'.txt') for i in range(self.num_data)]
        return graph_files

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        generator(self.num_data, self.raw_dir)

    def process(self):
        # Read data into huge `Data` list.
        data_list = load_data(os.path.join(self.raw_dir, "graph"), os.path.join(self.raw_dir, "label"), self.id_label)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

