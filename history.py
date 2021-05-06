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
        fig, ax = plt.subplots()  # Create a figure containing a single axes.
        ax.plot(range(len(self.train_loss)), self.train_loss, label="train")
        ax.plot([self.val_freq * x for x in range(len(self.valid_loss))], self.valid_loss, label="validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Avg Loss per graph")
        ax.set_title("Loss History")
        ax.legend()
        print("Save to loss_history.jpg")
        plt.savefig("loss_history.jpg")

        fig2, ax2 = plt.subplots()  # Create a figure containing a single axes.
        ax2.plot([self.val_freq * x for x in range(len(self.valid_acc))], self.valid_acc, label="validation")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Accuracy History")
        ax2.legend()
        print("Save to acc_history.jpg")
        plt.savefig("acc_history.jpg")
