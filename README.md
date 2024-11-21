```ini
[dataset]
file = datasets/bistro.hdf5
native-resolution = 225p     # which resolution to upsample from, specified using the frame height in pixels
target-resolution = 900p     # what is the final supersampling resolution
motion-mask-threshold = 1.0  # threshold for the motion mask, see nsrt.ipynb for details on how to set this value

[optimization]               # optimization options will be ignored if checkpoint is provided
optimizer = RangerAdaBelief  # options: RangerAdaBelief, AdaBelief, AdamW, SGD
learning-rate = 0            # use 0 for default learning rate (recommended for RangerAdaBelief and AdaBelief)

[training]
checkpoint = ""       # from which checkpoint to resume training
epochs = 400          # the number of epochs (including checkpoint's trained epochs) after which the training will stop
patch-size = 80       # training is performed on square patches cropped from the input frames, not the whole frame
spatial-stride = 60   # this defines the overlap between patches to crop from the input frames
chunk-size = 8        # for recurrent training, we load a chunk of consecutive frames to process
temporal-stride = 4   # this defines the overlap between chunks when extracting from a traning sequence
batch-size = 8        # a training batch contains batch-size chunks of consecutive patches
num-workers = 8       # number of workers for data loading

[model]
convo-features = 32   # number of intermediate feature maps accross convolutional layers
frame-channels = 10   # diffuse (3) + specular (3) + normal (3) + depth (1)
context-length = 3    # number of previous PLUS the current frame to consider for temporal context
```
