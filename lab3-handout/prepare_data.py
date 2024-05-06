import argparse
from data import find_images, ImageCollection
from typing import Tuple
from shutil import copy
import random
from os import makedirs
import os
import pathlib
import logging

parser = argparse.ArgumentParser(
                    prog='prepare',
                    description='Prepare data',
                    epilog='')

parser.add_argument('--train',type=float, required=False, default=0.6, help='train weight')
parser.add_argument('--val',type=float, required=False, default=0.2, help='val weight')
parser.add_argument('--test',type=float, required=False, default=0.2, help='test weight')
parser.add_argument('-i', '--input',type=pathlib.Path, required=True, help='input path')
parser.add_argument('-o', '--output',type=pathlib.Path, required=True, help='output path')


def split_collection(images: ImageCollection, train:float, val:float, test:float, seed=1667)->Tuple[ImageCollection,ImageCollection,ImageCollection]:
    # all sorting is done to make the collection creation deterministic

    # always the same shuffling
    rnd = random.Random(seed)
    
    trainset = {}
    valset   = {}
    testset  = {}

    for label in sorted(images.keys()):
        # sort for predictabibility
        imagepaths = sorted(images[label])
        rnd.shuffle(imagepaths)

        # normalization weight
        w = train + val + test

        valstart = round((train * len(imagepaths)) / w)
        teststart = valstart + round((val*len(imagepaths)) / w)

        trainset[label] = imagepaths[0:valstart]
        valset[label] = imagepaths[valstart:teststart]
        testset[label] = imagepaths[teststart:]

        logging.info("Label: %s has %d entries for train, %d for val, %d for test" % (label, len(trainset[label]), len(valset[label]), len(testset[label])))

    return trainset, valset, testset

def prepare(args):
    trainset, valset, testset = split_collection(find_images(args["input"]), train=args["train"], val=args["val"], test=args["test"])

    def copyall(fset, split, basepath):
        for label in fset:
            makedirs(os.path.join(basepath, split, label), exist_ok=True)
            for fpath in fset[label]:
                src = fpath
                srcname = os.path.basename(src)
                dst = os.path.join(basepath, split, label, srcname)

                print("Copy '%s' to '%s'" % (src, dst))
                copy(src, dst)

    makedirs(args["output"], exist_ok=False)

    logging.info("Copying train...")
    copyall(trainset, "train", args["output"])

    logging.info("Copying val...")
    copyall(valset, "val", args["output"])

    logging.info("Copying test...")
    copyall(testset, "test", args["output"])

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s | %(levelname)s | %(module)s | %(funcName)s:%(lineno)s ] %(message)s')
    print("hej")
    args = parser.parse_args()
    prepare(vars(args))