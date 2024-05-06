from torch import optim, nn
from infrastructure import Experiment
from model import YourFirstNet

class YourFirstCNN(Experiment):
    def init_model(self, n_labels, **kwargs):
        self.ckpt.model =YourFirstNet(n_labels)
        self.ckpt.criterion = nn.CrossEntropyLoss()
        self.ckpt.optimizer = optim.Adam(self.ckpt.model.parameters())