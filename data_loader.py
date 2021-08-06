import os
import re
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import numpy as np
from torch_geometric.utils import contains_self_loops
from torch_geometric.utils import degree


label_node_feature = {}
for i in range(0, 5):
    label_node_feature[i] = []

label_edge_feature = {}
for i in range(0, 5):
    label_edge_feature[i] = []

#node feature
# 0: asap 
# 1: in_degree 
# 2: out_degree 
# 3: no_grandparent  this is not used
# 4: no_grandchild  this is not used
# 5: no_ancestor 
# 6: no_descendant 
# 7: is_mem 

#edge feature
#0. number_nodes_in_between
#1. same_asap_node
#2. asap_diff
#3. ancestor_
#4. descendant
label_node_feature[0] = [ 0,1, 2 ,5, 6, 7]
label_node_feature[1] = [5,6]
label_node_feature[2] = []
label_node_feature[3] = [0, 5, 6]


#edge feature
label_edge_feature[2] = [0, 1, 2,3,4]
label_edge_feature[3] = [0, 1, 2, 3, 4]

label_to_hash_tag_index_ = {}
label_to_hash_tag_index_[0] = 0
label_to_hash_tag_index_[1] = 2
label_to_hash_tag_index_[2] = 3
label_to_hash_tag_index_[3] = 3


#this function is to get graph data for inference
def get_single_inference_graph_data(graph_dir,  file, label_id):
    data = Data()
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


    # Get feature data 
    feat_filename = file[:-4]+"_feature.txt"
    f_feat = open(os.path.join(graph_dir, feat_filename), 'r')
    x = []
    num_node = 0
    asap = {}
    num_ancestor = {}
    num_descendant = {}
    ## read feature
    same_level_nodes_edge_info = {}
    edge_feature = {}
    feature_file_hash_tag_id = 0
    for line in f_feat:
        if "###" in line:
            feature_file_hash_tag_id +=1
            continue

        if feature_file_hash_tag_id == 0:
            features = line.strip()
            features = features.split(" ")
            # print(features)
            int_features = []
            for s in features:
                int_features.append(int(s))
            final_features = []
            for index in label_node_feature[label_id]:
                final_features.append(int_features[index])

            asap[num_node] = int_features[0]
            num_ancestor [num_node] = int_features[5]
            num_descendant [num_node] = int_features[6]
                
            x.append(final_features)
            num_node += 1
            
        
        elif feature_file_hash_tag_id == 1:
            tmp = line.strip().split()
            tmp = list(map(int, tmp))
            same_level_nodes_edge_info[(tmp[0], tmp[1])] = tmp[2:]
            
        elif feature_file_hash_tag_id == 2:
            tmp = line.strip().split()
            tmp = list(map(int, tmp))
            edge_feature[str(tmp[0])+ "_" + str (tmp[1])] = tmp[2:]

        else:
            assert(False)
            

    x = torch.tensor(x, dtype=torch.long)
    f_feat.close()



    # print("filename", file)
    # Get label data
    if label_id == 0:  # Schedule Order 
        data = Data(x=x, edge_index=edge_index)
    # elif id_label == 1:
    #     new_edge_index = []
    #     new_edge_index.append([])
    #     new_edge_index.append([])
    #     edge_attr = []
    #     for i   in range(0, len(edge_index[0])):
    #         a_node = int(edge_index[0][i])
    #         b_node = int(edge_index[1][i])
    #         new_edge_index[0].append(a_node)
    #         new_edge_index[1].append(b_node)

    #         #undirectd edge
    #         new_edge_index[0].append(b_node)
    #         new_edge_index[1].append(a_node)
    #         asap_diff = abs (asap[a_node] - asap[b_node])
    #         edge_attr.append([asap_diff, num_ancestor[a_node] ])
    #         edge_attr.append([asap_diff, num_descendant[b_node] ])
    #     # print("new graph\n" , edge_index, edge_attr, y)
    #     new_edge_index = torch.tensor(new_edge_index, dtype=torch.long)
    #     edge_attr = torch.tensor(edge_attr, dtype=torch.long)
    #     data = Data(x=x, edge_index=new_edge_index, edge_attr = edge_attr)

    elif label_id == 1:  # Start Node Distance or Neighbour distance
        current_idx = 0  # Indicate which label is reading
        new_edge_index = []
        new_edge_index.append([])
        new_edge_index.append([])
        edge_attr = []
        y = []
        for node in same_level_nodes_edge_info.items():
            new_edge_index[0].append(node[0][0])
            new_edge_index[1].append(node[0][1])
            # print(node[1])
            edge_attr.append(node[1] )
                   
        # print("new graph\n" , edge_index, edge_attr, y)
        new_edge_index = torch.tensor(new_edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.long)
        # print(edge_attr)
        data = Data(x=x, edge_index=new_edge_index, edge_attr = edge_attr)
        
    elif label_id == 2:
        edge_attr = []
        for i in range(len(edge_index[0])):
            # print( str(int(edge_index[0][i])) +"_"+  str(int(edge_index[1][i]))  , edge_info[ str(int(edge_index[0][i])) +"_"+  str(int(edge_index[1][i])) ] )
            # print( str(int(edge_index[0][i])) +"_"+  str(int(edge_index[1][i] )) )
            temp_features = []
            int_edge_feature = edge_feature[str(int(edge_index[0][i])) +"_"+  str(int(edge_index[1][i] ))]
            for index in label_edge_feature[label_id]:
                temp_features.append(int_edge_feature[index])
            edge_attr.append(temp_features)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    elif label_id == 3:
        edge_attr = []
        # print("edge_feature", edge_feature)
        for i in range(len(edge_index[0])):
            # print( str(int(edge_index[0][i])) +"_"+  str(int(edge_index[1][i]))  , edge_info[ str(int(edge_index[0][i])) +"_"+  str(int(edge_index[1][i])) ] )
            # print( str(int(edge_index[0][i])) +"_"+  str(int(edge_index[1][i] )) )
            temp_features = []
            int_edge_feature = edge_feature[str(int(edge_index[0][i])) +"_"+  str(int(edge_index[1][i] ))]
            for index in label_edge_feature[label_id]:
                temp_features.append(int_edge_feature[index])
            edge_attr.append(temp_features)
        
        # print("new graph\n" , x, edge_index, y)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    else:
        assert(False)

    return data
   
# this function is to get data for training
def get_single_graph_data(graph_dir, label_dir,  file, label_id):
    data = Data()
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


    # Get feature data 
    feat_filename = file[:-4]+"_feature.txt"
    f_feat = open(os.path.join(graph_dir, feat_filename), 'r')
    x = []
    num_node = 0
    asap = {}
    num_ancestor = {}
    num_descendant = {}
    ## read feature
    # startnode_feature = False
    same_level_nodes_edge_info = {}
    edge_feature = {}
    feature_file_hash_tag_id = 0
    for line in f_feat:
        if "###" in line:
            feature_file_hash_tag_id +=1
            continue

        if feature_file_hash_tag_id == 0:
            features = line.strip()
            features = features.split(" ")
            # print(features)
            int_features = []
            for s in features:
                int_features.append(int(s))
            final_features = []
            for index in label_node_feature[label_id]:
                final_features.append(int_features[index])

            asap[num_node] = int_features[0]
            num_ancestor [num_node] = int_features[5]
            num_descendant [num_node] = int_features[6]
                
            x.append(final_features)
            num_node += 1
            
        
        elif feature_file_hash_tag_id == 1:
            tmp = line.strip().split()
            tmp = list(map(int, tmp))
            same_level_nodes_edge_info[(tmp[0], tmp[1])] = tmp[2:]
            
        elif feature_file_hash_tag_id == 2:
            tmp = line.strip().split()
            tmp = list(map(int, tmp))
            edge_feature[str(tmp[0])+ "_" + str (tmp[1])] = tmp[2:]

        else:
            assert(False)
            

    x = torch.tensor(x, dtype=torch.long)
    f_feat.close()



    # print("filename", file)
    # Get label data
    f_label = open(os.path.join(label_dir, file), 'r')
    if label_id == 0:  # Schedule Order or Communication Value
        y = np.zeros(num_node)  # Spare space for labels in case the node is out-of-order
        current_idx = 0  # Indicate which label is reading
        for line in f_label:
            # Jump if current line is a seperator line or the label is not wanted.
            if line == "###\n":
                current_idx += 1
                continue
            # Parse the wanted line for labels
            if current_idx == label_to_hash_tag_index_[label_id]:  # We need to
                a, b = line.strip().split()
                y[int(a)] = int(b)/2   # Should the id of node written in the label? NO.
        y = torch.tensor(y, dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, y=y)
    # elif id_label == 1:
    #     y = np.zeros(num_node)  # Spare space for labels in case the node is out-of-order
    #     current_idx = 0  # Indicate which label is reading
    #     for line in f_label:
    #         # Jump if current line is a seperator line or the label is not wanted.
    #         if line == "###\n":
    #             current_idx += 1
    #             continue
    #         # Parse the wanted line for labels
    #         if current_idx == id_label:  # We need to
    #             a, b = line.strip().split()
    #             y[int(a)] = int(b)  # Should the id of node written in the label? NO.
    #     y = torch.tensor(y, dtype=torch.float)
    #     new_edge_index = []
    #     new_edge_index.append([])
    #     new_edge_index.append([])
    #     edge_attr = []
    #     for i   in range(0, len(edge_index[0])):
    #         a_node = int(edge_index[0][i])
    #         b_node = int(edge_index[1][i])
    #         new_edge_index[0].append(a_node)
    #         new_edge_index[1].append(b_node)

    #         #undirectd edge
    #         new_edge_index[0].append(b_node)
    #         new_edge_index[1].append(a_node)
    #         asap_diff = abs (asap[a_node] - asap[b_node])
    #         edge_attr.append([asap_diff, num_ancestor[a_node] ])
    #         edge_attr.append([asap_diff, num_descendant[b_node] ])
    #     # print("new graph\n" , edge_index, edge_attr, y)
    #     new_edge_index = torch.tensor(new_edge_index, dtype=torch.long)
    #     edge_attr = torch.tensor(edge_attr, dtype=torch.long)
    #     data = Data(x=x, edge_index=new_edge_index, edge_attr = edge_attr, y=y)

    elif label_id == 1:  # Start Node Distance or Neighbour distance
        current_idx = 0  # Indicate which label is reading
        new_edge_index = []
        new_edge_index.append([])
        new_edge_index.append([])
        edge_attr = []
        y = []
        for line in f_label:
            # Jump if current line is a seperator line or the label is not wanted.
            if line == "###\n":
                current_idx += 1
                continue
            # Parse the wanted line for labels
            if current_idx == label_to_hash_tag_index_[label_id]:
                tmp = line.strip().split()
                tmp = list(map(int, tmp))
                if (tmp[0], tmp[1]) in same_level_nodes_edge_info:
                    new_edge_index[0].append(tmp[0])
                    new_edge_index[1].append(tmp[1])
                    edge_attr.append(same_level_nodes_edge_info[(tmp[0], tmp[1])])
                    y.append(tmp[2])

                elif (tmp[1], tmp[0]) in same_level_nodes_edge_info:
                    new_edge_index[0].append(tmp[1])
                    new_edge_index[1].append(tmp[0])
                    edge_attr.append(same_level_nodes_edge_info[(tmp[1], tmp[0])])
                    y.append(tmp[2])
                else:
                    assert(False)
        # print("new graph\n" , edge_index, edge_attr, y)
        new_edge_index = torch.tensor(new_edge_index, dtype=torch.long)
        edge_attr = torch.tensor(edge_attr, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.float)
        data = Data(x=x, edge_index=new_edge_index, edge_attr = edge_attr, y=y)
    elif label_id == 2:
        # neigbor distance
        current_idx = 0  # Indicate which label is reading
        edge_label_info = {}
        edge_label_num = 0
        for line in f_label:
            # Jump if current line is a seperator line or the label is not wanted.
            if line == "###\n":
                current_idx += 1
                continue
            # Parse the wanted line for labels
            if current_idx == label_to_hash_tag_index_[label_id]:
                edge_label_num +=1 
                tmp = line.strip().split()
                edge_label_info[str(tmp[0])+ "_" + str (tmp[1]) ] =  tmp[2]
        y = []
        edge_attr = []
        if edge_label_num == 0:
            return None
        # print("edge_info", edge_info)
        for i in range(len(edge_index[0])):
            # print( str(int(edge_index[0][i])) +"_"+  str(int(edge_index[1][i]))  , edge_info[ str(int(edge_index[0][i])) +"_"+  str(int(edge_index[1][i])) ] )
            value = (int(edge_label_info[ str(int(edge_index[0][i])) +"_"+  str(int(edge_index[1][i])) ] ))
            y.append(value)
            assert(str(int(edge_index[0][i])) +"_"+  str(int(edge_index[1][i] )) in edge_feature.keys())
            temp_features = []
            int_edge_feature = edge_feature[str(int(edge_index[0][i])) +"_"+  str(int(edge_index[1][i] ))]
            for index in label_edge_feature[label_id]:
                temp_features.append(int_edge_feature[index])
            edge_attr.append(temp_features)
           
        
        # print("new graph\n" , x, edge_index, y)
        y = torch.tensor(y, dtype=torch.float)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        data = Data(x=x, edge_index=edge_index,  edge_attr=edge_attr, y=y)
    elif label_id == 3:
        # neigbor distance
        current_idx = 0  # Indicate which label is reading
        edge_label_info = {}
        edge_label_num = 0
        for line in f_label:
            # Jump if current line is a seperator line or the label is not wanted.
            if line == "###\n":
                current_idx += 1
                continue
            # Parse the wanted line for labels
            if current_idx == label_to_hash_tag_index_[label_id]:
                edge_label_num +=1 
                tmp = line.strip().split()
                edge_label_info[str(tmp[0])+ "_" + str (tmp[1]) ] =  tmp[3]
        y = []
        edge_attr = []
        if edge_label_num == 0:
            return None
        # print("edge_feature", edge_feature)
        for i in range(len(edge_index[0])):
            # print( str(int(edge_index[0][i])) +"_"+  str(int(edge_index[1][i]))  , edge_info[ str(int(edge_index[0][i])) +"_"+  str(int(edge_index[1][i])) ] )
            y.append( (int(edge_label_info[ str(int(edge_index[0][i])) +"_"+  str(int(edge_index[1][i])) ] )) )
            # print( str(int(edge_index[0][i])) +"_"+  str(int(edge_index[1][i] )) )
            temp_features = []
            int_edge_feature = edge_feature[str(int(edge_index[0][i])) +"_"+  str(int(edge_index[1][i] ))]
            for index in label_edge_feature[label_id]:
                temp_features.append(int_edge_feature[index])
            edge_attr.append(temp_features)
        
        # print("new graph\n" , x, edge_index, y)
        y = torch.tensor(y, dtype=torch.float)
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    else:
        print("wrong label", label_id)
        assert(False)

    f_label.close()
    return data
    
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
        data = get_single_graph_data(graph_dir, label_dir, file, id_label)
        if data is not None:
            dataset.append(data)
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

