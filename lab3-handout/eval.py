import argparse
import torch
from torch.utils.data import DataLoader
from os import makedirs
from data import ExperimentDataset
from sklearn.metrics import classification_report
from tqdm import tqdm
import pathlib
from matplotlib import pyplot as plt
import logging
from yaml import load, CLoader as Loader

from infrastructure import Experiment, load_yaml
import experiments # otherwise experiments will not be known

parser = argparse.ArgumentParser(
                    prog='eval',
                    description='Evaluate model',
                    epilog='')

parser.add_argument('-i', '--input',type=pathlib.Path, required=True, help='input config')
parser.add_argument('--dataset',type=str, default="test", help='which dataset to evalute')
parser.add_argument('--ckpt',type=pathlib.Path, required=True, help='checkpoint path')
parser.add_argument('--output',type=pathlib.Path, required=True, help='eval output path')
parser.add_argument('--batchsize',type=int, default=32, help='batchsize')
parser.add_argument('--device',type=str, default="cpu", help='device')
parser.add_argument('--raw', action="store_true", default=False, help='plot raw unsmoothed data only')
parser.add_argument('--num-dl-workers',type=int, default=1, help='num dataloader workers (converting images on the fly)')

def create_dataloader(args):
    options = load_yaml(args["input"])
    ds = ExperimentDataset.create(options[args["dataset"]]["dataset"])

    pin_options = {}
    if args["device"] == "cuda":
        pin_options["pin_memory"] = True
        pin_options["pin_memory_device"] = "cuda"

    dl_eval = DataLoader(ds, batch_size=args["batchsize"], num_workers=args["num_dl_workers"], **pin_options)
    return dl_eval

def evaluate(args):
    experiment = Experiment.fromfile(args["ckpt"])

    dl_eval = create_dataloader(args)
    y_true, y_pred = [], []
    
    device = torch.device(args["device"])

    experiment.to(args["device"])
    experiment.eval()
    i = 0
    for batch in tqdm(dl_eval):
        X, y_true_batch = batch
        X, y_true_batch = X.to(device), y_true_batch.to(device)

        y_pred_batch = experiment.predict(X).argmax(dim=1)

        y_true.append(y_true_batch.cpu())
        y_pred.append(y_pred_batch.cpu())
        
    
    y_true, y_pred = torch.hstack(y_true), torch.hstack(y_pred)
    print(f"Y_true: {y_true.shape}")
    print(f"y_pred: {y_pred.shape}")
    makedirs(args["output"], exist_ok=True)
    print(y_pred)
    with open(args["output"] / "eval-report.txt", "w") as fout:
        fout.write(classification_report(y_true=y_true,y_pred=y_pred))
        


    # Plot learning cruve
    plt.figure(figsize=(8,4))
    plt.title("Learning curve")
    plt.grid(True)

    if args["raw"]:
        iteration, loss = zip(*enumerate(experiment.metrics.train_loss.raw_history))
        val_iteration, val_loss = zip(*enumerate(experiment.metrics.val_loss.raw_history))
        plt.plot(iteration, loss, 'b', label='Raw train')
        plt.plot(val_iteration, val_loss, 'r', label='Raw val')
    else:
        iteration, loss = zip(*enumerate(experiment.metrics.train_loss.history))
        val_iteration, val_loss = zip(*enumerate(experiment.metrics.val_loss.history))
        plt.plot(iteration, loss, 'b', label="Train")
        plt.plot(val_iteration, val_loss, 'r', label="Val")
        plt.plot(experiment.metrics.epoch_iters.history, experiment.metrics.train_epoch_loss.history, 'go--', linewidth=2, markersize=8, alpha=0.5, label="Train epoch")
        plt.plot(experiment.metrics.epoch_iters.history, experiment.metrics.val_epoch_loss.history, 'mo--', linewidth=2, markersize=8, alpha=0.5, label="Val epoch")
    plt.legend()
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.savefig(args["output"] / "learningcurve.pdf")
    
    # Plot accuracy curve
    plt.figure(figsize=(8,4))
    plt.title("Accuracy")
    plt.grid(True)
    if args["raw"]:
        iteration, acc = zip(*enumerate(experiment.metrics.train_acc.raw_history))
        val_iteration, val_acc = zip(*enumerate(experiment.metrics.val_acc.raw_history))
        plt.plot(iteration, acc, 'b', label='Raw train')
        plt.plot(val_iteration, val_acc, 'r', label='Raw val')
    else:
        iteration, acc = zip(*enumerate(experiment.metrics.train_acc.history))
        val_iteration, val_acc = zip(*enumerate(experiment.metrics.val_acc.history))
        plt.plot(iteration, acc, 'b', label="Train")
        plt.plot(val_iteration, val_acc, 'r', label="Val")
        plt.plot(experiment.metrics.epoch_iters.history, experiment.metrics.train_epoch_acc.history, 'go--', linewidth=2, markersize=8, alpha=0.5, label="Train epoch")
        plt.plot(experiment.metrics.epoch_iters.history, experiment.metrics.val_epoch_acc.history, 'mo--', linewidth=2, markersize=8, alpha=0.5, label="Val epoch")
    plt.legend()
    plt.xlabel("iteration")
    plt.ylabel("accuracy")
    plt.savefig(args["output"] / "accuracy.pdf")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d ] %(message)s')
    args = parser.parse_args()
    evaluate(vars(args))