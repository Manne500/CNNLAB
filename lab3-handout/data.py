import os
import glob
import torch
import torch.jit
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import v2
import torchvision.transforms as T
from typing import Dict, List, Tuple, NewType, Any
from tqdm import tqdm
import logging
from torchvision.models import resnet50, ResNet50_Weights

ImageCollection = NewType("ImageCollection", Dict[str,List[str]])

def find_images(path)->ImageCollection:
    label2images = {}
    for imagepath in glob.glob(str(path) + "/*/*.jpg"):
        label = os.path.basename(os.path.dirname(imagepath))
        if os.path.isfile(imagepath):
            label2images.setdefault(label, []).append(imagepath)
    
    return label2images

class ByteToFloat(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, X):
        return X.float() / 255.

# Base-class to find all datasets
class ExperimentDataset(Dataset):
    @staticmethod
    def create(options: Dict[str, Any])->Dataset:
        type = options["type"]

        def get_subclasses(cls):
            for subclass in cls.__subclasses__():
                yield from get_subclasses(subclass)
                yield subclass

        for cls in get_subclasses(ExperimentDataset):
            if cls.__name__ == type:
                return cls(**options)
        
        raise ValueError("Could not find experiment dataset type: %s, candidates are: %s" % (type, ",  ".join([cls.__name__ for cls in get_subclasses()])))

class ImageFolderDataset(datasets.ImageFolder, ExperimentDataset):
    """Dataset that loads images from directory"""
    def create_transforms(self, res=224, augment="none", **kwargs):

        to_tensor = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True)
        ])


        if augment==1:
            print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAHHHHH")
            fn = v2.Compose([
            T.TrivialAugmentWide(),
            ResNet50_Weights.IMAGENET1K_V2.transforms()
            ])
        else:
            fn = v2.Compose([
            to_tensor,
            v2.Resize(res),
            v2.CenterCrop(res),
            ])



        return fn

    def __init__(self, path, **kwargs) -> None:
        datasets.ImageFolder.__init__(self, path, self.create_transforms(**kwargs))
        ExperimentDataset.__init__(self)
        logging.info("Found %d images", len(self))


class ImagePreloadFolderDataset(ExperimentDataset):
    """Dataset that preloads all images to memory"""
    def __init__(self, path, repeat=0, res=224, **kwargs):
        """Construct this image dataset
        
        Parameters
        ----------
        repeat : int
            The number of times the same images should be repeated (most useful for image augmentation)
        """
        super().__init__()
        folder_ds = ImageFolderDataset(path, res=res, **kwargs)
        num_images = len(folder_ds)*(repeat+1)

        # The most efficient way to preallocate the memory for all images
        self.Xs = torch.empty((num_images,3,res,res), dtype=torch.float32)
        self.ys = torch.empty((num_images,), dtype=torch.long)

        # Copy the classes found
        self.classes = folder_ds.classes

        k = 0
        with tqdm(desc="Preloading images", total=num_images) as pbar:
            for i in range(len(folder_ds)):
                for r in range(repeat+1):
                    X, y = folder_ds[i]
                    self.Xs[k] = X
                    self.ys[k] = y
                    k += 1
                    pbar.update(1)

        logging.info("Preloaded %d images", len(self))

    def __len__(self):
        return len(self.Xs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.Xs[index], self.ys[index]
