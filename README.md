
# use example:
First, activate conda environment:
> conda  activate lisa
## Training:
Training for all the labels.  
>`` run_training.sh  test ``  
>
Note: "test" is the data folder name in "data", and wiil also be used as the model name. The program will save model as "test.pt".

## Inference:
Command line:
> `` python gnn_inference.py 0VIHB3KrVi test``
> 
"0VIHB3KrVi" is the graph name and corresponding graph info files are in folder "data/infer". "test" is the model we previously generated. 

