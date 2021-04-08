import os
import re
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import numpy as np
from torch_geometric.utils import contains_self_loops
from torch_geometric.utils import degree

def load_data(graph_dir, label_dir, id_label, graph_files):
    """
    Function:
        Load data set into a list of torch_geometric.data.Data objects
    Params:
        graph_dir: str, directory to the graph (edge) information,
        label_dir: str, directory to the label information,
        id_label: int, index of label.
    Return:
        list of torch_geometric.data.Data objects
    """
    dataset = []
    for file in graph_files:
        # Get edge data
        edge = []
        f_graph = open(os.path.join(graph_dir, file), 'r')
        for line in f_graph:
            n_out, n_in = line.strip().split()
            edge.append([int(n_out), int(n_in)])
        edge = torch.tensor(edge, dtype=torch.long)
        edge_index = edge.t().contiguous()
        if contains_self_loops(edge_index):
            print(edge_index)
            assert False
        f_graph.close()

        # Get feature data - ASAP
        feat_filename = file[:-4]+"_feature.txt"
        f_feat = open(os.path.join(graph_dir, feat_filename), 'r')
        x = []
        num_node = 0
        for line in f_feat:
            num_node += 1
            asap_value = line.strip()
            x.append([int(asap_value)])
        x = torch.tensor(x, dtype=torch.long)
        f_feat.close()

        # Get label data
        f_label = open(os.path.join(label_dir, file), 'r')
        if id_label in [0, 1]:  # Schedule Order or Communication Value
            y = np.zeros(num_node)  # Spare space for labels in case the node is out-of-order
            current_idx = 0  # Indicate which label is reading
            for line in f_label:
                # Jump if current line is a seperator line or the label is not wanted.
                if line == "###\n":
                    current_idx += 1
                    continue
                # Parse the wanted line for labels
                if current_idx == id_label:  # We need to
                    a, b = line.strip().split()
                    y[int(a)] = int(b)  # Should the id of node written in the label? NO.
            print(y)
            y = torch.tensor(y, dtype=torch.float)
            data = Data(x=x, edge_index=edge_index, y=y)
            dataset.append(data)

        else:  # Start Node Distance or Neighbour distance
            current_idx = 0  # Indicate which label is reading
            for line in f_label:
                # Jump if current line is a seperator line or the label is not wanted.
                if line == "###\n":
                    current_idx += 1
                    continue
                # Parse the wanted line for labels
                if current_idx == id_label:
                    tmp = line.strip().split()
                    y = [int(x) for x in tmp]  # Use the graph information to predict only one line of data
                    print(y)
                    y = torch.tensor(y, dtype=torch.float)
                    data = Data(x=x, edge_index=edge_index, y=y)
                    dataset.append(data)
        f_label.close()
    return dataset

class dfg_dataset(InMemoryDataset):

    def __init__(self, root, id_labels, transform=None, pre_transform=None):
        """

        :param root: str, root of dataset
        :param id_labels: list of int, id of feature to use
        :param transform:
        :param pre_transform:
        """
        self.id_labels = id_labels
        super(dfg_dataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        graph_files = os.listdir(os.path.join(self.raw_dir, "label"))  # Available graph data are those with labels, so use label file name as available graph file name
        graph_files = [x for x in graph_files if x[-3:] == "txt"]
        return graph_files

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        return 0

    def process(self):
        # Read data into huge `Data` list.
        data_list = load_data(os.path.join(self.raw_dir, "graph"), os.path.join(self.raw_dir, "label"), self.id_labels, self.raw_file_names)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

