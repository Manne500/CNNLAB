from torch import optim, nn
from infrastructure import Experiment
from model import *

class YourFirstCNN(Experiment):
    def init_model(self, n_labels, **kwargs):
        self.ckpt.model =YourFirstNet(n_labels)
        self.ckpt.criterion = nn.CrossEntropyLoss()
        self.ckpt.optimizer = optim.Adam(self.ckpt.model.parameters())


class SimpleCNN(Experiment):
    def init_model(self, n_labels,dropout, **kwargs):
        self.ckpt.model =SimpleSeqCNN(n_labels,dropout)
        self.ckpt.criterion = nn.CrossEntropyLoss()
        self.ckpt.optimizer = optim.Adam(self.ckpt.model.parameters())
class ResNetJr_exp(Experiment):
    def init_model(self, n_labels,dropout, **kwargs):
        self.ckpt.model =ResNetJr(n_labels,dropout)
        self.ckpt.criterion = nn.CrossEntropyLoss()
        self.ckpt.optimizer = optim.Adam(self.ckpt.model.parameters())