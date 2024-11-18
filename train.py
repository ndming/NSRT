from argparse import ArgumentParser
from configparser import ConfigParser

from model.data import HDF5Dataset, get_dataloader
from model.nsrt import NSRT
from model.loss import Criterion
from model.step import Trainer, Validator

from utils import get_optimizer

def main():
    # Get training configurations
    args = get_argument_parser().parse_args()
    config = ConfigParser()
    config.read(args.config)

    dataset = HDF5Dataset(config.get('dataset', 'file'))
    train_dataloader, val_dataloader = get_dataloader(dataset, config)

    model = NSRT()

    optimizer, scheduler = get_optimizer(config, model)
    criterion = Criterion()

    trainer = Trainer(model, train_dataloader, optimizer, criterion)
    validator = Validator(model, val_dataloader, criterion)

    # Keep training until the total number of epochs is reached
    n_epochs = config.getint('training', 'epochs')
    while scheduler.last_epoch < n_epochs:
        trainer.step()
        scheduler.step()
        validator.step()


def get_argument_parser():
    parser = ArgumentParser(description="Train the NSRT model")

    parser.add_argument(
        '--config', type=str, required=True, metavar='FILE',
        help="which file to read training configurations from")
    parser.add_argument(
        '--seed', type=int, default=0, 
        help="random seed for reproducibility")

    return parser

if __name__ == "__main__":
    main()