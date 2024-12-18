import torch
import torch.utils.data as data
import numpy as np

from pathlib import Path
import h5py

class HDF5Dataset(data.Dataset):
    def __init__(
            self, file, native_resolution, target_resolution, motion_threshold,
            patch_size, spatial_stride, chunk_size, temporal_stride, 
            train=True, on_detect_nan=None):
        super().__init__()
        self.path = Path(file)
        self.hdf5 = h5py.File(file, 'r')

        self.native_resolution = native_resolution
        self.target_resolution = target_resolution
        self.upsampling_factor = int(target_resolution[:-1]) // int(native_resolution[:-1])  # 900p / 225p = 4
        self.motion_threshold  = motion_threshold  # save this for Trainer and Validator

        self.patch_size = patch_size
        self.spatial_stride = spatial_stride
        self.chunk_size = chunk_size
        self.temporal_stride = temporal_stride

        self.train = train
        self.on_detect_nan = on_detect_nan

        n_patches_width, n_patches_height = self._count_patches()
        self.patches_per_frame   = n_patches_width * n_patches_height if train else 1
        self.chunks_per_sequence = (self.hdf5.attrs['frames-per-sequence'] - chunk_size) // temporal_stride + 1
        self.frames_per_sequence = self.hdf5.attrs['frames-per-sequence']

    def __len__(self):
        sequence_count = self.hdf5.attrs['train-sequences'] if self.train else self.hdf5.attrs['test-sequences']
        return self.patches_per_frame * self.chunks_per_sequence * sequence_count
    
    def __getitem__(self, idx):
        samples_per_sequence = self.patches_per_frame * self.chunks_per_sequence
        seq_idx   = idx // samples_per_sequence
        remaining = idx %  samples_per_sequence
        chunk_idx = remaining // self.patches_per_frame
        patch_idx = remaining %  self.patches_per_frame

        s_digits = self.hdf5.attrs['sequence-index-digits']
        f_digits = self.hdf5.attrs['frame-index-digits']

        native_chunk = []
        target_chunk = []
        for i in range(self.chunk_size):
            set = 'train' if self.train else 'test'
            frame_idx = chunk_idx * self.temporal_stride + i
            native_frame = self.hdf5[f'{self.native_resolution}/{set}/seq-{seq_idx:0{s_digits}d}/frame-{frame_idx:0{f_digits}d}']
            target_frame = self.hdf5[f'{self.target_resolution}/{set}/seq-{seq_idx:0{s_digits}d}/frame-{frame_idx:0{f_digits}d}']
            native_frame = self._populate_native_frame(native_frame, patch_idx)  # (14, H, W)
            target_frame = self._populate_target_frame(target_frame, patch_idx)  # ( 8, H, W)

            if torch.isnan(native_frame).any() or torch.isinf(native_frame).any():
                if self.on_detect_nan:
                    self.on_detect_nan(seq_idx, frame_idx, self.train, self.native_resolution)
                native_frame = torch.nan_to_num(native_frame, nan=0.0, posinf=1.0, neginf=0.0)
            
            if torch.isnan(target_frame).any() or torch.isinf(target_frame).any():
                if self.on_detect_nan:
                    self.on_detect_nan(seq_idx, frame_idx, self.train, self.target_resolution)
                target_frame = torch.nan_to_num(target_frame, nan=0.0, posinf=1.0, neginf=0.0)

            native_chunk.append(native_frame)
            target_chunk.append(target_frame)

        return torch.stack(native_chunk, dim=0), torch.stack(target_chunk, dim=0)

    def _populate_native_frame(self, frame, patch_idx):
        # Convert to numpy arrays first, otherwise the loading would be extremely slow
        diffuse  = torch.tensor(np.array(frame['diffuse-dir']) + np.array(frame['diffuse-ind']))  # (3, H, W)
        specular = torch.tensor(np.array(frame['glossy-dir'])  + np.array(frame['glossy-ind']))   # (3, H, W)
        normal   = torch.tensor(np.array(frame['normal']))                                        # (3, H, W)
        albedo   = torch.tensor(np.array(frame['diffuse-col']) + np.array(frame['glossy-col']))   # (3, H, W)
        rough    = torch.tensor(np.array(frame['roughness'])).unsqueeze(0)                        # (1, H, W)
        vector   = torch.tensor(np.array(frame['vector']))                                        # (4, H, W)

        # Fix the vector values so that the 2 first components map pixels in the current frame to the previous frame,
        # and the 2 last components map pixels in the current frame to the next frame
        # This step purely depends on the nature of the dataset, and may not be necessary for other datasets
        # For this dataset, the y-components must be flipped, and the 2 last components must be negated
        vector[1,  ...] = -vector[1,  ...]
        vector[3,  ...] = -vector[3,  ...]
        vector[2:, ...] = -vector[2:, ...]

        # Tonemap HDR lighting components
        diffuse  = tonemap(diffuse)
        specular = tonemap(specular)

        # Training is done on patches
        if self.train:
            diffuse  = self._crop_patch(diffuse,  patch_idx)
            specular = self._crop_patch(specular, patch_idx)
            normal   = self._crop_patch(normal,   patch_idx)
            albedo   = self._crop_patch(albedo,   patch_idx)
            rough    = self._crop_patch(rough,    patch_idx)
            vector   = self._crop_patch(vector,   patch_idx)

        frame = torch.cat([diffuse, specular, normal, albedo, rough, vector], dim=0)
        return frame
    
    def _populate_target_frame(self, frame, patch_idx):
        # Convert to numpy arrays first, otherwise the loading would be extremely slow
        diffuse  = torch.tensor(np.array(frame['diffuse-dir']) + np.array(frame['diffuse-ind']))  # (3, H, W)
        specular = torch.tensor(np.array(frame['glossy-dir'])  + np.array(frame['glossy-ind']))   # (3, H, W)
        diff_col = torch.tensor(np.array(frame['diffuse-col']))                                   # (3, H, W)
        spec_col = torch.tensor(np.array(frame['glossy-col']))                                    # (3, H, W)
        vector   = torch.tensor(np.array(frame['vector']))[:2, ...]                               # (2, H, W)

        vector[1,  ...] = -vector[1,  ...]  # flip the y-component
        diffuse  = tonemap(diffuse)
        specular = tonemap(specular)

        # Training is done on patches
        if self.train:
            diffuse  = self._crop_patch(diffuse,  patch_idx, self.upsampling_factor)
            specular = self._crop_patch(specular, patch_idx, self.upsampling_factor)
            diff_col = self._crop_patch(diff_col, patch_idx, self.upsampling_factor)
            spec_col = self._crop_patch(spec_col, patch_idx, self.upsampling_factor)
            vector   = self._crop_patch(vector,   patch_idx, self.upsampling_factor)

        frame = torch.cat([diffuse, specular, diff_col, spec_col, vector], dim=0)
        return frame
    
    def _crop_patch(self, tensor, patch_idx, factor=1):
        n_patches_width, _ = self._count_patches()
        x = (patch_idx %  n_patches_width) * self.spatial_stride * factor
        y = (patch_idx // n_patches_width) * self.spatial_stride * factor
        return tensor[..., y:y + self.patch_size * factor, x:x + self.patch_size * factor]
    
    def _count_patches(self):
        # Count how many patches we can stride in each spatial dimension
        width, height = self.hdf5[self.native_resolution].attrs['frame-width'], self.hdf5[self.native_resolution].attrs['frame-height']
        n_patches_along_width  = (width  - self.patch_size) // self.spatial_stride + 1
        n_patches_along_height = (height - self.patch_size) // self.spatial_stride + 1
        return n_patches_along_width, n_patches_along_height


def get_train_loaders(dataset, batch_size, n_workers, split=None):
    if split is None:
        indices = torch.arange(len(dataset))
        indices = indices[torch.randperm(len(indices))]
        train_size = int(0.8 * len(indices))
        train_indices, val_indices = indices[:train_size], indices[train_size:]
    else:
        train_indices, val_indices = split
        train_indices = train_indices[torch.randperm(len(train_indices))]
        val_indices = val_indices[torch.randperm(len(val_indices))]
    
    train_sampler = data.SubsetRandomSampler(train_indices)
    val_sampler   = data.SubsetRandomSampler(val_indices)
    train_loader  = data.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, pin_memory=True, num_workers=n_workers)
    val_loader    = data.DataLoader(dataset, batch_size=batch_size, sampler=val_sampler,   pin_memory=True, num_workers=n_workers)
    return train_loader, val_loader, train_indices, val_indices


def tonemap(x, gamma=2.2):
    # See Equation (1) in https://doi.org/10.1145/3543870
    x = torch.clamp(x, 0, 65535)
    x = torch.log(x + 1).pow(1 / gamma)
    return x
