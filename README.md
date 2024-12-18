# Neural Supersampling for Real-time Rendering with Enhanced Radiance Demodulation

## Environment
Model implementation uses Pytorch. Training and testing are guaranteed to run on Linux with:
- Pytorch `2.5.1`
- CUDA `12.1`

See `requirements.txt` for additional required package.

## Dataset
The current implementation use HDF5 dataset files, but effort are being made to make this approach plausible for
distributed data training with multi-process.

The structure of the HDF5 source is expected to have the following hierarchy:

```
/
|
|____225p
|   |
|   |____train
|   |   |
|   |   |____seq-000
|   |   |   |
|   |   |   |____frame-000
|   |   |   |   |
|   |   |   |   |____combined
|   |   |   |   |
|   |   |   |   |____diffuse-col
|   |   |   |   |
|   |   |   |   |____diffuse-dir
|   |   |   |   |
|   |   |   |   |____diffuse-ind
|   |   |   |   |
|   |   |   |   |____glossy-dir
|   |   |   |   |
|   |   |   |   |____glossy-ind
|   |   |   |   |
|   |   |   |   |____glossy-col
|   |   |   |   |
|   |   |   |   |____depth
|   |   |   |   |
|   |   |   |   |____normal
|   |   |   |   |
|   |   |   |   |____roughness
|   |   |   |   |
|   |   |   |   |____vector
|   |   |   |
|   |   |   |____frame-001
|   |   |   |
|   |   |   |____frame-...
|   |   |   |
|   |   |   |____frame-099
|   |   |
|   |   |____seq-001
|   |   |
|   |   |____seq-...
|   |   |
|   |   |____seq-015
|   |
|   |____test
|       |
|       |____seq-000
|       |
|       |____seq-...
|       |
|       |____seq-003
|
|____900p
    |
    |____train
    |
    |____test
```

See [model/data.py](https://github.com/ndming/NSRT/blob/main/model/data.py) for more details.

## Training

```
python train.py --config path/to/train.ini
```

### Training configuraions
Find below the detailed descriptions of each config option.

### dataset
- `native-resolution`: which resolution to upsample from, specified using the frame height in pixels (i.e. `1080p`).
- `target-resolution`: the final supersampling resolution.
- `motion-mask-threshold`: threshold for the motion mask. This value might need to be tunned based on the nature of the
motion vectors generated in the dataset.

### optimization
Optimization options will be ignored if checkpoint is provided.
- `optimizer`: options: `RangerAdaBelief`, `AdaBelief`, `AdamW`, `SGD`.
- `learning-rate`: use 0 for default learning rate (recommended for `RangerAdaBelief` and `AdaBelief`).

### training
- `checkpoint`: from which checkpoint to resume training. If the provided path checkpoint file is not exists, the training
will save the checkpoint at the specified path. An empty path will result in a randomly generated checkpoint name.
- `epoches`: the number of training epochs after which the training will stop. Note that this also counts the number of 
epoches that have been trained in the checkpoint if one exists.
- `patch-size`: training is performed on square patches cropped from the input frames, not the whole frame.
- `spatial-stride`: this will result in overlapping between patches to crop from the input frames. Smaller patch strides 
increase the number of training examples.
- `chunk-size`: each training example consists of a chunk of consecutive frames. This defines how many frames to include
in the chunk.
- `temporal-stride`: this defines the overlap between chunks when extracting from a traning sequence. Smaller chunk
strides increase the number of training examples.
- `batch-size`: a training batch will have batch-size chunks of consecutive patches.

### loss
- `spatial-weight`: the spatial weight used in the loss computation, must be in `[0, 1]`.
- `vgg-depth`: the current implementation use perceptual loss from VGG16's feature extraction layers. This number defines
how many VGG16 layers will be used to compute the spatial loss.

### model
- `convo-features`: the number of intermediate feature maps accross convolutional layers.
- `frame-channels`: the number of channels in a frame when inputs are fed to the network. If the network is trained 
using diffuse, specular, normal, albedo, and depth maps, the number of channels to specify is 13.
- `context-length`: the number of previous frames PLUS the current frame to consider for temporal context.

### A training config example
```ini
[dataset]
file = datasets/bistro.hdf5
native-resolution = 225p     # which resolution to upsample from
target-resolution = 900p     # what is the final supersampling resolution
motion-mask-threshold = 1.0  # threshold for the motion mask

[optimization]               # will be ignored if checkpoint is provided
optimizer = RangerAdaBelief  # options: RangerAdaBelief, AdaBelief, AdamW, SGD
learning-rate = 0            # use 0 for default learning rate

[training]
checkpoint = ""       # from which checkpoint to resume training
epochs = 400          # the number of training epochs
patch-size = 80       # training is performed on square patches
spatial-stride = 60   # overlap degree between patches
chunk-size = 8        # the number of consecutive frames to process
temporal-stride = 4   # overlap degree between chunks
batch-size = 8        # the number of training examples in each training batch

[loss]
spatial-weight = 0.4  # loss spatial weight, must be in [0, 1]
vgg-depth = 9         # the number of VGG layers to use for spatial loss

[model]
convo-features = 32   # the number of intermediate feature maps
frame-channels = 10   # diffuse (3) + specular (3) + normal (3) + depth (1)
context-length = 3    # number of previous PLUS the current frame
```
