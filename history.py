import matplotlib.pyplot as plt
from torch import Tensor

allow_lisa_print= False



def lisa_print(input: Tensor, name = ""):
    if allow_lisa_print:
        print( name , input.size(), input)
class history():
    def __init__(self, val_freq = 20):
        self.train_loss = []
        self.valid_loss = []
        self.valid_acc = []
        self.best_val_acc = 0
        self.val_freq = val_freq
    def add_tl(self, loss):  # add train loss (average for each graph, so that the batch size doesn't matter)
        self.train_loss.append(loss)
        # Don't save the model during training, otherwise too many "saving" at the beginning

    def add_vl(self, loss):  # add validation loss
        self.valid_loss.append(loss)

    def add_valid_acc(self, acc):
        self.valid_acc.append(acc)
        if acc >= self.best_val_acc:
            self.best_val_acc = acc
            return True
        else:
            return False

    def plot_hist(self):
        return 0 
