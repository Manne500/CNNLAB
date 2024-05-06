import torch
from torch import nn, optim
from torch.utils import data
import pathlib
import os
from tqdm import tqdm
import pickle
from typing import Tuple, Optional, Dict, Any
from data import ExperimentDataset
from dataclasses import dataclass
from yaml import load, CLoader as Loader, dump, CDumper as Dumper
from torch.utils.data import Dataset, DataLoader
import logging

def accuracy(y_logits: torch.Tensor, y_true: torch.Tensor)-> Tuple[float, int]:
    """Compute accuracy
    
    Parameters
    ----------
    y_logits : torch.Tensor
        The raw output from the model, i.e. not softmaxed, typically of shape: (batch, num_classes)
    y_true: torch.Tensor
        The correct class index for each instance in y_logits, typically of shape (batch)

    Returns
    -------
    Tuple of (accuracy, correct count)

    accuracy : float
        The share of predictions that was correct
    correct count : int
        The exact count
    """
    y_pred = y_logits.softmax(dim=1).argmax(dim=1)
    num_correct = torch.count_nonzero(y_pred == y_true)
    return (num_correct / y_true.shape[0]).item(), num_correct.item()

def load_yaml(path):
    """Load a yaml file"""
    with open(path, "rb") as fin:
        return load(fin, Loader=Loader)

class RawValue:
    """Raw value with history"""
    def __init__(self)->None:
        self._history = []
    
    @property
    def value(self):
        return self._history[-1]
    
    @value.setter
    def value(self, v):
        self._history.append(v)

    @property
    def history(self):
        return self._history

    def __getstate__(self):
        return {"history": self.history}
    
    def __setstate__(self, state):
        self._history = state["history"]
    
class EMAValue:
    """Exponentially Moving Averaged Value
    
    Stores history of smoothed (averaged) values and raw
    """
    def __init__(self, halflife=8) -> None:
        self._halflife = halflife
        self._alpha = 0.5 ** (1.0/halflife)
        self._x = None
        self._ema_history = []
        self._raw_history = []
    
    @property
    def value(self):
        return self._x
       
    @value.setter
    def value(self, v):
        self._raw_history.append(v)

        if self._x is None:
            self._x = v
        else:
            self._x = self._x * self._alpha + (1.0 - self._alpha) * v
        self._ema_history.append(self._x)

    @property
    def history(self):
        return self._ema_history
    
    @property
    def raw_history(self):
        self._raw_history

    def __getstate__(self):
        return {"ema": self._ema_history, "raw": self._raw_history, "halflife": self._halflife, "x": self._x}
    
    def __setstate__(self, state):
        self._halflife = state["halflife"]
        self._alpha = 0.5 ** (1.0/self._halflife)
        self._x = state["x"]
        self._ema_history = state["ema"]
        self._raw_history = state["raw"]

@dataclass
class ExperimentMetrics:
    epoch_iters: RawValue = None

    train_loss: EMAValue = None
    train_acc: EMAValue = None
    train_epoch_loss: RawValue = None
    train_epoch_acc: RawValue = None

    val_loss: EMAValue = None
    val_acc: EMAValue = None
    val_epoch_loss: RawValue = None
    val_epoch_acc: RawValue = None

@dataclass
class ExperimentCheckpoint:
    # Your pytorch model
    model: nn.Module = None

    # Your optimizer
    optimizer: optim.Optimizer = None

    # Your loss function
    criterion: nn.Module = None

@dataclass
class ExperimentData:
    train: data.DataLoader = None
    val: Optional[data.DataLoader] = None

class Experiment(object):
    def __init__(self) -> None:
        self.metrics = ExperimentMetrics()
        self.ckpt = ExperimentCheckpoint()
        self.data = ExperimentData()

    def to(self, device):
        """"Move the experiment to a torch device"""
        self.device = torch.device(device)
        self.ckpt.model.to(self.device)
    
    def init(self, options: Dict[str, Any]):
        """Initialize the experiment
        
        Parameters
        ----------
        options: Dict[str, Any]
            experiment options, these are propagated:
             * metrics to init_metrics
             * model to init_model
        """
        self.options = options
        self.init_metrics(**options.get("metrics",{}))
        self.init_model(**options.get("model",{}))

    def init_metrics(self, halflife:int=8, **kwargs):
        # Epoch metrics
        self.metrics.epoch_iters = RawValue()
        
        self.metrics.train_epoch_acc = RawValue()
        self.metrics.train_epoch_loss = RawValue()

        self.metrics.val_epoch_acc = RawValue()
        self.metrics.val_epoch_loss = RawValue()

        # Per batch metrics
        self.metrics.train_loss = EMAValue(halflife)
        self.metrics.val_loss = EMAValue(halflife)

        self.metrics.train_acc = EMAValue(halflife)
        self.metrics.val_acc = EMAValue(halflife)

    def init_model(self, **kwargs):
        """Abstract method: Initialize the checkpoint (ckpt) with all that is needed to run training (model, criterion, and optimizer)"""
        raise NotImplementedError()

    def create_dataloader(self, 
                          dataset:data.Dataset, 
                          batchsize:int, 
                          num_workers:int=1, 
                          shuffle=True, 
                          **kwargs):
        """Create a dataloader for a dataset
        
        Parameters
        ----------
        dataset: data.Dataset
            The dataset to use for the datalaoder
        batchsize: int
            Size of the batches that the dataloader should provide
        num_workers: int
            The number of CPU threads/processes to use for preparing batches in advance
        **kwargs:
            Unhandled options are collected here
        """
        if self.device.type == "cuda":
            pin_memory=True
            pin_memory_device=str(self.device)
        else:
            pin_memory=False
            pin_memory_device=""

        return DataLoader(dataset, batch_size=batchsize, shuffle=shuffle, 
                          num_workers=num_workers, pin_memory=pin_memory, 
                          pin_memory_device=pin_memory_device, persistent_workers=True)

    def init_dataloader(self, options: Dict[str, Any]):
        """Initialize the dataloaders for training
        
        Parameters
        ----------
        options : Dict[str, Any]
            configuration options
        """

        # 1. Initialize a clean data object
        self.data = ExperimentData()
        
        # 2. Get batchsize or use default of 32
        batchsize = options.get("model", {}).get("batchsize", 32)
        logging.info("Using batchsize of %d", batchsize)

        # 3. Check if train exists in the options or raise error
        if options.get("train", {}).get("dataset", {}).get("path", None) is None:
            raise ValueError("Train dataset cannot be find, train.dataset.path in config does not exist!")

        # 4. Load training data
        logging.info("Using training data at %s", options["train"]["dataset"]["path"])
        train_ds = ExperimentDataset.create(options["train"]["dataset"])
        self.data.train = self.create_dataloader(train_ds, batchsize, **options["train"].get("dataloader", {}))

        # 5. Load validation data (optional)
        if options.get("val", {}).get("dataset", {}).get("path", None) is not None:
            logging.info("Using validation data at %s", options["val"]["dataset"]["path"])
            val_ds = ExperimentDataset.create(options["val"]["dataset"])
            self.data.val = self.create_dataloader(val_ds, batchsize, **options["val"].get("dataloader", {}))
    
    @staticmethod
    def factory(name: str) -> Optional["Experiment"]:
        """Used to find and construct an Experiment from a given name"""
        def get_subclasses(cls):
            for subclass in cls.__subclasses__():
                yield from get_subclasses(subclass)
                yield subclass

        for cls in get_subclasses(Experiment):
            if cls.__name__ == name:
                return cls()
        
        raise ValueError("Could not find experiment: %s, candidates are: %s" % 
                         (name, ",  ".join([cls.__name__ for cls in get_subclasses(Experiment)])))

    @staticmethod
    def fromfile(path: pathlib.Path, resume_options: Optional[pathlib.Path]=None):
        """Construct experiment form a file path"""
        model_ckpt = path / "model.ckpt"
        model_metadata = path / "model-metadata.yml"
        model_metrics = path / "model-metrics.pkl"

        # 1. Load metadata
        logging.info("Loading metadata...")
        options = load_yaml(model_metadata)

        if resume_options is not None:
            resume_options = load_yaml(resume_options, "r")
            options.update(resume_options)

        # 2. Construct experiment class
        logging.info("Initializing model...")
        instance = Experiment.factory(options["experiment"])

        # 3. Build to initialize empty model
        instance.init(options)

        # 4. Load weights into model
        logging.info("Loading weights...")
        instance.load_model_weights(torch.load(str(model_ckpt)))

        # 5. Load metrics
        logging.info("Loading metrics...")
        instance.metrics = pickle.load(open(model_metrics, "rb"))

        logging.info("Loading complete.")
        return instance

    def load_model_weights(self, state_dict):
        """Load model weights and state into the experiment"""
        self.ckpt.model.load_state_dict(state_dict["model"])
        self.ckpt.optimizer.load_state_dict(state_dict["optimizer"])
        self.ckpt.criterion.load_state_dict(state_dict["criterion"])

    def save_model_weights(self, path: pathlib.Path):
        """Save model weights and state to file for the experiment"""
        torch.save({
            "model": self.ckpt.model.state_dict(),
            "optimizer": self.ckpt.optimizer.state_dict(),
            "criterion": self.ckpt.criterion.state_dict()
        }, path)

    def save(self, path: pathlib.Path):
        """Save the experiment to disk with all weights and state"""
        os.makedirs(path, exist_ok=True)
        model_ckpt = path / "model.ckpt"
        model_options = path / "model-metadata.yml"
        model_metrics = path / "model-metrics.pkl"

        # 1. Save model weights and state
        self.save_model_weights(model_ckpt)

        # 2. Save experiment configuration
        with open(model_options, "w") as fout:
            dump(self.options, fout)
        
        # 3. Save metrics for the experiment
        with open(model_metrics, "wb") as fout:
            pickle.dump(self.metrics, fout)

    def predict(self, X: torch.Tensor):
        """Predict for sample X
        
        Run the forward pass of the model and apply softmax to produce predictions.
        """
        # TODO: Write your code here
        m = nn.Softmax(dim=1)
        return m(self.ckpt.model(X))
      

    def eval(self):
        """Put the experiment and model into eval mode"""
        self.ckpt.model.eval()

    def infinite_batches(self, dl):
        """Given a dataloader instance create an infinite generator of batches"""
        while True:
            for b in dl:
                yield b

    def train(self):
        self.init_dataloader(self.options)

        if self.data.train is None:
            raise ValueError("Train dataloder is not initialized, got None!")

        if self.ckpt.model is None:
            raise ValueError("Model is not initialized!")

        # Used for resume, stores the number of iterations completed or initialize to 0
        self.options["__iterations"] = self.options.get("__iterations", 0)
        self.options["__epoch"] = self.options.get("model").get("__epoch", 0)

        epochs = self.options.get("model", {}).get("epochs", 0)

        # Validation is usually never larger than training, 
        # as it is just an estimate loop it for all eternity
        val_batches = self.infinite_batches(self.data.val) if self.data.val is not None else None

        iteration = self.options["__iterations"]
        try:
            for epoch in range(self.options["__epoch"], epochs):
                epoch_iterations = 0
                
                # train accumlators
                train_epoch_loss_sum = 0.0
                train_epoch_acc_sum = 0
                
                # val accumlators
                val_epoch_loss_sum = 0.0
                val_epoch_acc_sum = 0

                # num instances seen
                train_instances = 0
                val_instances = 0

                logging.info("Epoch %i/%i" % (epoch+1, epochs))
                
                self.on_epoch_start(epoch)
                with tqdm(self.data.train) as train_batches:
                    for train_batch in train_batches:
                        self.ckpt.model.train()
                        
                        # Training data
                        X_train, y_train = train_batch
                        X_train, y_train = X_train.to(self.device), y_train.to(self.device)
                        train_instances += X_train.shape[0]

                        # Take one training step
                        train_loss, train_logits = self.step(X_train, y_train, epoch, iteration)
                        train_epoch_loss_sum += train_loss * X_train.shape[0]
                        
                        # Add train loss and accuracy to our metrics
                        self.metrics.train_loss.value = train_loss
                        self.metrics.train_acc.value, train_num_correct  = accuracy(train_logits, y_train)
                        train_epoch_acc_sum += train_num_correct
                        
                        # If validation data exists, evaluate one batch
                        if val_batches is not None:
                            # Get a validation data
                            X_val, y_val = next(val_batches)
                            X_val, y_val = X_val.to(self.device), y_val.to(self.device)

                            val_instances += X_val.shape[0]

                            # Compute validation minibatch loss and accuracy
                            with torch.no_grad():
                                self.ckpt.model.eval()
                                val_logits = self.ckpt.model(X_val)
                                val_loss = self.ckpt.criterion(val_logits, y_val).item()

                                self.metrics.val_loss.value = val_loss
                                self.metrics.val_acc.value, val_num_correct  = accuracy(val_logits, y_val)

                                val_epoch_loss_sum += val_loss * X_val.shape[0]
                                val_epoch_acc_sum += val_num_correct
                        
                            # Set the loss into our progressbar, avoid refreshing it for performance
                            train_batches.set_postfix({"loss": "%.04f" % self.metrics.train_loss.value, 
                                                       "val_loss": "%.04f" % self.metrics.val_loss.value, 
                                                       "acc":  "%.03f" % self.metrics.train_acc.value, 
                                                       "val_acc":  "%.03f" % self.metrics.val_acc.value}, 
                                                       refresh=False)
                        else:
                            train_batches.set_postfix({"loss": "%.04f" % self.metrics.train_loss.value, 
                                                       "acc":  "%.03f" % self.metrics.train_acc.value}, 
                                                       refresh=False)

                        iteration += 1
                        epoch_iterations += 1

                    self.options["__iterations"] = iteration

                # Save the iteration count for current epoch, can be used to track epochs in metrics
                self.metrics.epoch_iters.value = iteration

                # Protect against division by zero
                if train_instances == 0:
                    train_instances = 1

                if val_instances == 0:
                    val_instances = 1

                # Update epoch loss values (global average)
                self.metrics.train_epoch_loss.value = train_epoch_loss_sum / train_instances
                self.metrics.val_epoch_loss.value = val_epoch_loss_sum / val_instances

                # Update epoch accuracy values (average for all examples)
                self.metrics.train_epoch_acc.value = train_epoch_acc_sum / train_instances
                self.metrics.val_epoch_acc.value = val_epoch_acc_sum / val_instances

                # Print it out for status and record keeping
                logging.info("Train loss (average): %.5f, acc (average): %.4f" % 
                             (self.metrics.train_epoch_loss.value, self.metrics.train_epoch_acc.value))
                
                logging.info("Val loss (average): %.5f, acc (average): %.4f" % 
                             (self.metrics.val_epoch_loss.value, self.metrics.val_epoch_acc.value))
                
                if not self.on_epoch_finish(epoch):
                    logging.info("Training stopped as requested.")
                    break
        except KeyboardInterrupt:
            # This allows for a clean abort that saves current model to take place.
            logging.warn("Training aborted.")
    
    def on_epoch_start(self, epoch: int):
        """Event hook, called before each epoch starts"""
        pass

    def on_epoch_finish(self, epoch: int)->bool:
        """Event hook, called after each epoch is finished
        
        Returns
        -------
        bool
            indicate if training should continue
        """
        return True

    def step(self, X: torch.Tensor, y: torch.Tensor, epoch: int, iteration: int)->Tuple[float, torch.Tensor]:
        """Make one training-step.

           Return: train_loss and logits (raw values from the last layer)
        """
        #TODO implement changed learning rate m8b3?
        self.ckpt.optimizer.zero_grad()
        outputs = self.ckpt.model(X)
        loss = self.ckpt.criterion(outputs.softmax(dim=1), y)
        loss.backward()
        self.ckpt.optimizer.step()
        return loss,outputs

