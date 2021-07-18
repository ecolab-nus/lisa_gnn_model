import sys, os
import pathlib
from torch_geometric.data import Data
sys.path.append('../dfg_generator/dfg')
sys.path.append('../dfg_generator')
sys.path.append('../dfg_generator/graph_generation')
from util import get_graph
from data_loader import get_single_inference_graph_data
from Label0.lisa import label0_inference
from Label1.lisa import label1_inference
from Label2.lisa import label2_inference
from Label3.lisa import label3_inference
from Label4.lisa import label4_inference


graph_name =  sys.argv[1]
infer_model_name = "final_model"
if len(sys.argv) != 3:
    assert(False)

infer_model_name =  sys.argv[2]

graph_file_name = graph_name + ".txt"
print("graph_file_name", graph_file_name)
graph_path = pathlib.Path().absolute().parent
graph_path = os.path.join(graph_path, "data/infer")
get_graph(graph_file_name , graph_path, graph_path )


label_file =  open(graph_path + "/" + graph_name + "_label.txt", "w")

data = get_single_inference_graph_data(graph_path, graph_file_name, 0)
value = label0_inference(data, infer_model_name)
value = value.tolist()
print("label0", value)
for i in range(len(value)):
    label_file.write(str(i) + " "+ str(int(value[i])) + "\n")
label_file.write("###\n")


# data = get_single_inference_graph_data(graph_path, graph_file_name, 1)
# value = label1_inference(data, infer_model_name)
# value = value.tolist()
# print("label1", value)
# for i in range(len(value)):
#     label_file.write(str(i) + " "+ str(int(value[i])) + "\n")
# label_file.write("###\n")

data = get_single_inference_graph_data(graph_path, graph_file_name, 2)
value = label2_inference(data, infer_model_name)
value = value.tolist()
print("label2", value)
for i in range(len(value)):
    label_file.write(str(int(data.edge_index[0][i]))+ " " + str(int(data.edge_index[1][i])) + " "+ str(int(value[i])) + "\n")
label_file.write("###\n")


data = get_single_inference_graph_data(graph_path, graph_file_name, 3)
value = label3_inference(data, infer_model_name)
value = value.tolist()
print("label3", value)
for i in range(len(value)):
    label_file.write(str(int(data.edge_index[0][i]))+ " " + str(int(data.edge_index[1][i])) + " "+ str(int(value[i])) + "\n")

label_file.write("###\n")
data = get_single_inference_graph_data(graph_path, graph_file_name, 4)
value = label4_inference(data, infer_model_name)
value = value.tolist()
print("label4", value)
for i in range(len(value)):
    label_file.write(str(int(data.edge_index[0][i]))+ " " + str(int(data.edge_index[1][i])) + " "+ str(int(value[i])) + "\n")

label_file.close()

