from torch import optim, nn
from infrastructure import Experiment
from model import *
from torchvision.models import resnet50, ResNet50_Weights

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
class RealResNet50(Experiment):
    def init_model(self, n_labels,dropout, finetune, **kwargs):
        self.ckpt.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.ckpt.criterion = nn.CrossEntropyLoss()
        print(f"Finetune: {finetune}")
        for p in self.ckpt.model.parameters():
            p.requires_grad_(finetune)
        self.ckpt.optimizer = optim.Adam(self.ckpt.model.parameters())
        self.ckpt.model.fc = nn.Linear(2048, 5)
