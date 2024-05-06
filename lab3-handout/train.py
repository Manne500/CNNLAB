import os
import argparse
import pathlib
import logging
from typing import Dict, Any, Optional

def train(config: pathlib.Path, 
          ckpt: pathlib.Path, 
          device: str, 
          resume=False,
          resume_ckpt: Optional[pathlib.Path]=None, 
          do_sanity_check=False):
    """Training method"""

    # Just to speed up the start of train
    from infrastructure import Experiment, load_yaml
    import experiments # otherwise experiments will not be known

    if resume:
        if resume_ckpt is not None:
            experiment = Experiment.fromfile(resume_ckpt, config)
        else:
            experiment = Experiment.fromfile(ckpt, config)
    else:
        options = load_yaml(config) # type: Dict[str, Any]
        experiment = Experiment.factory(options["experiment"])
        experiment.init(options)

        if do_sanity_check:
            logging.info("Saving and loading model to check for bugs...")
            experiment.save(ckpt)
            Experiment.fromfile(ckpt)
            logging.info("Sanity check passed")

    experiment.to(device)
    experiment.train()

    logging.info("Train completed, saving model...")
    experiment.save(ckpt)
    logging.info("Done.")

def main(args):
    logging.debug(args)

    if not args["resume"]:
        os.makedirs(args["ckpt"], exist_ok=args["force"])
    elif not os.path.exists(args["ckpt"]):
        logging.critical("Could not find checkpoint directory '%s', unable to resume!", args["ckpt"])
        os.exit(1)

    train(args["config"], 
          args["ckpt"], 
          args["device"], 
          do_sanity_check=not args["no_sanity_check"], 
          resume=args["resume"], 
          resume_ckpt=args.get("resume_ckpt", None))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                        prog='train',
                        description='Trains a flower model',
                        epilog='')

    # Parser arguments
    parser.add_argument('--config',type=pathlib.Path, required=True, help='config path')
    parser.add_argument('--resume-ckpt',type=pathlib.Path, required=False, help='optional, resume checkpoint directory, if diffrent from ckpt')
    parser.add_argument('--ckpt',type=pathlib.Path, required=True, help='checkpoint directory, where to store the result')
    parser.add_argument('-r', "--resume", default=False, action="store_true", help='resume training, i.e. load before training')
    parser.add_argument('-f', '--force', default=False, action="store_true", help='overwrite training data if it already exists')
    parser.add_argument('-d', '--device', type=str, default="cpu", help="device to use: cpu/cuda")
    parser.add_argument('-s', '--no-sanity-check', default=False, action="store_true", help="avoid sanity check (save and load empty model)")

    # Logging settings
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d ] %(message)s')
    args = parser.parse_args()
    main(vars(args))