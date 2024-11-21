from argparse import ArgumentParser
from configparser import ConfigParser

import os, torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP

from model.data import HDF5Dataset, get_train_loaders
from model.nsrt import NSRT
from model.loss import Criterion
from model.step import Trainer, Validator

from utils import get_optimizer, write_inference, write_checkpoint, gen_id
from pathlib import Path

from rich.progress import Progress, BarColumn, TextColumn

def main(rank, world_size, config, n_dataload_workers):
    # Initialize the distributed process group
    if world_size > 1:
        setup(rank, world_size)

    dataset = HDF5Dataset(
        file=config.get('dataset', 'file'),
        native_resolution=config.get('dataset', 'native-resolution'),
        target_resolution=config.get('dataset', 'target-resolution'),
        motion_threshold=config.getfloat('dataset', 'motion-mask-threshold'),
        patch_size=config.getint('training', 'patch-size'),
        chunk_size=config.getint('training', 'chunk-size'),
        spatial_stride=config.getint('training', 'spatial-stride'),
        temporal_stride=config.getint('training', 'temporal-stride'),
        train=True
    )

    model = NSRT(
        frame_channels=config.getint('model', 'frame-channels'),
        context_length=config.getint('model', 'context-length'),
        upscale_factor=dataset.upsampling_factor,
        conv_features=config.getint('model', 'convo-features')
    )

    optimizer, scheduler = get_optimizer(config, model)
    criterion = Criterion(rank)

    checkpoint_path = config.get('training', 'checkpoint')
    checkpoint_path = Path(f"checkpoints/{dataset.path.stem}-{gen_id()}.pth") if not checkpoint_path else Path(checkpoint_path)
    
    split = None
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(checkpoint['model_latest'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        split = checkpoint['train_indices'], checkpoint['val_indices']
        print(f"Resuming training from epoch {checkpoint['epoch'] + 1}")

    if world_size > 1:
        model = DDP(model.to(rank), device_ids=[rank])
    elif world_size == 1:
        model = DataParallel(model).to(rank)

    # Keep training until the total number of epochs is reached
    batch_size=config.getint('training', 'batch-size')
    n_epochs = config.getint('training', 'epochs')
    while scheduler.last_epoch < n_epochs:
        train_loader, val_loader, train_indices, val_indices = get_train_loaders(dataset, batch_size, n_dataload_workers, split)
        batch_count = len(train_loader)

        trainer = Trainer(rank, model, train_loader, optimizer, criterion)
        validator = Validator(rank, model, val_loader, criterion)

        if world_size < 2 or rank == 0:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                TextColumn(" [progress.percentage]{task.percentage:>3.0f}% "),
                BarColumn(),
                TextColumn(f" [{{task.completed:>{len(str(batch_count))}}}/{{task.total}}] |"),
                TextColumn("train loss: {task.fields[loss]:.6f}"),
            ) as progress:
                task = progress.add_task(f"[cyan]Epoch {scheduler.last_epoch + 1}/{n_epochs}", total=batch_count, loss=float('inf'))
                on_loss_update = lambda _, loss: progress.update(task, advance=1, loss=loss)
                avg_train_loss = trainer.step(on_loss_update)

                scheduler.step()

                on_metrics_update = lambda _, loss, ssim, psnr: progress.console.print(f"Validating: loss - {loss:.4f} | SSIM - {ssim:.2f} | PSNR - {psnr:.2f}", end="\r")
                avg_loss, avg_ssim, avg_psnr, best_logits, best_target = validator.step(on_metrics_update)
                progress.console.print(f"[green]Validation: avg. loss - {avg_loss:.4f} | avg. ssim - {avg_ssim:.2f} | psnr. - {avg_psnr:.2f}")

                write_checkpoint(
                    model, optimizer, scheduler, scheduler.last_epoch, avg_train_loss, avg_loss, 
                    train_indices, val_indices, checkpoint_path)

                output_path = Path(f"checkpoints/{dataset.path.stem}/epoch-{scheduler.last_epoch + 1:03d}.exr")
                write_inference(best_logits, best_target, output_path)


        else:
            trainer.step()
            scheduler.step()
            validator.step()

    # Cleanup the distributed process group
    if world_size > 1:
        cleanup()


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    

def cleanup():
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = ArgumentParser(description="Train the NSRT model")
    parser.add_argument(
        '--config', type=str, required=True, metavar='FILE',
        help="which file to read training configurations from")
    parser.add_argument(
        '--seed', type=int, default=0, 
        help="random seed for reproducibility")
    parser.add_argument(
        '--n-gpus', type=int, default=1, 
        help="number of CUDA devices to use")
    
    # Get training configurations
    args = parser.parse_args()
    config = ConfigParser()
    config.read(args.config)

    # Perform minor config verification
    if config.getint('model', 'context-length') - 1 > config.getint('training', 'chunk-size'):
        print("[!] The chunk-size is less than context-length, leading to redundant computations")
        exit(1)

    # Set the random seed for reproducibility
    if args.seed:
        torch.manual_seed(args.seed)

    world_size = args.n_gpus
    if world_size > 1:
        n_workers = 0  # multiprocessing cause issues with HDF5 file access
        mp.spawn(main, args=(world_size, config, n_workers), nprocs=world_size, join=True)
    else:
        n_workers = config.getint('training', 'num-workers')
        main("cuda" if world_size == 1 else "cpu", world_size, config, n_workers)