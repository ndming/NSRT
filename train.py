from argparse import ArgumentParser
from configparser import ConfigParser
from pathlib import Path

import os, torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP

from model.data import HDF5Dataset, get_train_loaders
from model.nsrt import NSRT
from model.loss import Criterion
from model.step import Trainer, Validator
from utils import get_optimizer, write_inference, write_checkpoint, gen_id, unpack_model_state

from rich.progress import Progress, BarColumn, TextColumn
from rich.console import Console

console = Console()

def main(rank, world_size, config, n_workers):
    # world_size = number of GPUs (0: CPU, 1: Single GPU (DataParallel), >1: Multi-GPU (DistributedDataParallel))
    # rank will be the GPU index if world_size > 2, otherwise it will be either "cuda", "mps" or "cpu"

    # Initialize the distributed process group if using DDP
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
    split = None  # will be intialized if a checkpoint file at the path is found

    if checkpoint_path.exists():
        # Load all internal tensors to CPU, we wil move them to the correct device later
        checkpoint = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
        model = init_model(checkpoint, model, rank)
        
        # Init optimizer and scheduler
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

        # Get the split indices for the training and validation sets
        split = checkpoint['train_indices'], checkpoint['val_indices']
        console.print(f"Resuming training from epoch [cyan]{scheduler.last_epoch + 1}")
    else:
        # Simnply move the model to the correct device if there is no checkpoint
        if world_size > 1:
            model = DDP(model.to(rank), device_ids=[rank])
        elif world_size == 1:
            model = DataParallel(model).to(rank)

    # Keep training until the total number of epochs is reached
    batch_size = config.getint('training', 'batch-size')
    n_epochs   = config.getint('training', 'epochs')
    while scheduler.last_epoch < n_epochs:
        # We initialize train dataloaders at every epoch iteration to shuffle the training and validation indices
        train_loader, val_loader, train_indices, val_indices = get_train_loaders(dataset, batch_size, n_workers, split)

        spatial_weight = config.getfloat('training', 'spatial-weight')
        trainer   = Trainer(rank, model, train_loader, optimizer, criterion, spatial_weight)
        validator = Validator(rank, model, val_loader, criterion, spatial_weight)

        train_batch_count = len(train_loader)
        val_batch_count   = len(val_loader)

        # We can safely print to the console if we are the master process or the number of processes is 1
        if world_size < 2 or rank == 0:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                TextColumn(" [progress.percentage]{task.percentage:>3.0f}% "),
                BarColumn(bar_width=20),
                TextColumn(f" [{{task.completed:>{len(str(train_batch_count))}}}/{{task.total}}]"),
                TextColumn("> train loss: curr. {task.fields[loss]:.4f} | avg. {task.fields[avg_loss]:.4f}", style="gray"),
                console=console
            ) as progress:
                # Training will update the progress bar
                task = progress.add_task(f"[cyan]Epoch {scheduler.last_epoch + 1}/{n_epochs}", total=train_batch_count, loss=float('inf'), avg_loss=float('inf'))
                on_loss_update = lambda _, loss, avg_loss: progress.update(task, advance=1, loss=loss, avg_loss=avg_loss)
                avg_train_loss = trainer.step(on_loss_update)
            
            # Decay the learning rate
            scheduler.step()

            # Update the validation metrics at each validation batch
            on_metrics_update = lambda batch, loss, ssim, psnr: console.print(
                f"Testing [{batch + 1:>{len(str(val_batch_count))}}/{val_batch_count}]: loss - {loss:.4f} | SSIM - {ssim:.2f} | PSNR - {psnr:.2f}", end="\r")
            avg_loss, avg_ssim, avg_psnr, best_logits, best_target = validator.step(on_metrics_update)

            # Print the final average validation metrics
            console.print(f"Validation: avg. loss - {avg_loss:.4f} | avg. ssim - {avg_ssim:.2f} | psnr. - {avg_psnr:.2f}")

            write_checkpoint(
                model, optimizer, scheduler, avg_train_loss, avg_loss, avg_ssim, avg_psnr,
                train_indices, val_indices, checkpoint_path)
            
            # Save best inference results
            output_path = Path(f"checkpoints/{checkpoint_path.stem}/epoch-{scheduler.last_epoch:03d}.exr")
            write_inference(best_logits, best_target, output_path)
        
        # Child-process training path, will be developed in the future
        else:
            trainer.step()
            scheduler.step()
            validator.step()

    # Cleanup the distributed process group
    if world_size > 1:
        cleanup()


def init_model(checkpoint, model, rank):
    # Init the model first if training was OR being done on CPU
    if checkpoint['cpu_checkpoint']:
        model.load_state_dict(checkpoint['model_latest'])
    elif world_size < 1:
        # Unpack the model if it was saved under the DDP/DataParallel module
        model.load_state_dict(unpack_model_state(checkpoint['model_latest']))

    # Otherwise, wrap the model with the right parallel module
    if world_size > 1:
        if not checkpoint['ddp_checkpoint']:
            console.print(f"[yellow] Checkpoint was trainined with DataParallel, it won't be guaranteed to work properly with DDP")
        model = DDP(model.to(rank), device_ids=[rank])
    elif world_size == 1:
        if checkpoint['ddp_checkpoint']:
            console.print(f"[yellow] Checkpoint was trainined with DDP, it won't be guaranteed to work properly with DataParallel")
        model = DataParallel(model).to(rank)

    # The CPU training path has been properly initialized
    if world_size > 0:
        model.load_state_dict(checkpoint['model_latest'])


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    

def cleanup():
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = ArgumentParser(description="Train the NSRT model")
    parser.add_argument(
        '-c', '--config', type=str, required=True, metavar='',
        help="which file to read training configurations from")
    parser.add_argument(
        '-s', '--seed', type=int, default=0, metavar='',
        help="random seed for reproducibility")
    parser.add_argument(
        '-d', '--n-gpus', type=int, default=1, metavar='',
        help="number of GPU devices to use, 0 for CPU")
    parser.add_argument(
        '-n', '--num-workers', type=int, default=8, metavar='',
        help="number of workers for data loading")
    
    # Get training configurations
    args = parser.parse_args()
    config = ConfigParser()
    config.read(args.config)

    # Perform minor config verification
    if config.getint('model', 'context-length') - 1 > config.getint('training', 'chunk-size'):
        console.print("[red]The specified chunk-size is less than context-length, leading to redundant computations")
        exit(1)

    # Set the random seed for reproducibility
    if args.seed:
        torch.manual_seed(args.seed)

    world_size = args.n_gpus
    try:
        if world_size > 1:
            console.print(f"[yellow]Multiprocessing with GPUs is under development, falling back to single GPU")
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            main(device, 1, config, args.num_workers)
    
            # TODO: multiprocessing cause issues with HDF5 file access
            # mp.spawn(main, args=(world_size, config, n_workers), nprocs=world_size, join=True)
        else:
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            main(device if world_size == 1 else "cpu", world_size, config, args.num_workers)
    except KeyboardInterrupt:
        console.print("[yellow]Exit training due to keyboard interrupt, checkpoint is saved for the last finished epoch")