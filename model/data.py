import torch
import torch.utils.data as data

from pathlib import Path

import h5py

class HDF5Dataset(data.Dataset):
    def __init__(self, file, native_resolution, target_resolution, patch_size, spatial_stride, chunk_size, temporal_stride, train=True):
        super().__init__()
        file_path = Path(file)
        self.name = file_path.stem
        self.hdf5 = h5py.File(file, 'r')

        self.native_resolution = native_resolution
        self.target_resolution = target_resolution
        self.upsampling_factor = int(target_resolution[:-1]) // int(native_resolution[:-1])  # 900p / 225p = 4

        self.patch_size = patch_size
        self.spatial_stride = spatial_stride
        self.chunk_size = chunk_size
        self.temporal_stride = temporal_stride
        self.train = train

        n_patches_width, n_patches_height = self._count_patches()
        self.patches_per_frame   = n_patches_width * n_patches_height if train else 1
        self.chunks_per_sequence = (self.hdf5.attrs['frames-per-sequence'] - chunk_size) // temporal_stride + 1
        self.frames_per_sequence = self.hdf5.attrs['frames-per-sequence']
        self.channels_per_frame  = 3 + 3 + 3 + 1 + 2  # diffuse, specular, normal, depth, vector

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
            native_frame = self.hdf5[f'{self.native_resolution}/{set}/seq-{seq_idx:0{s_digits}d}/frame-{chunk_idx * self.temporal_stride + i:0{f_digits}d}']
            target_frame = self.hdf5[f'{self.target_resolution}/{set}/seq-{seq_idx:0{s_digits}d}/frame-{chunk_idx * self.temporal_stride + i:0{f_digits}d}']
            native_chunk.append(self._populate_frame(native_frame, patch_idx, factor=1))
            target_chunk.append(self._populate_frame(target_frame, patch_idx, factor=self.upsampling_factor))

        return torch.stack(native_chunk, dim=0), torch.stack(target_chunk, dim=0)

    def _populate_frame(self, frame, patch_idx, factor):
        diffuse  = torch.tensor(frame['diffuse-dir'] + frame['diffuse-ind'])  # (3, H, W)
        specular = torch.tensor(frame['glossy-dir']  + frame['glossy-ind'])   # (3, H, W)
        normal   = torch.tensor(frame['normal'])                              # (3, H, W)
        depth    = torch.tensor(frame['depth']).unsqueeze(0)                  # (1, H, W)
        vector   = torch.tensor(frame['vector'][:2, :, :])                    # (2, H, W)

        # Tonemap light components
        diffuse  = tonemap(diffuse)
        specular = tonemap(specular)

        # Training is done on patches
        if self.train:
            diffuse  = self._crop_patch(diffuse,  patch_idx, factor)
            specular = self._crop_patch(specular, patch_idx, factor)
            normal   = self._crop_patch(normal,   patch_idx, factor)
            depth    = self._crop_patch(depth,    patch_idx, factor)
            vector   = self._crop_patch(vector,   patch_idx, factor)

        frame = torch.cat([diffuse, specular, normal, depth, vector], dim=0)
        return frame
    
    def _crop_patch(self, tensor, patch_idx, factor):
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
    
    def __del__(self):
        self.hdf5.close()


def tonemap(x, gamma=2.2):
    # See Equation (1) in https://doi.org/10.1145/3543870
    x = torch.clamp(x, 0, 65535)
    x = torch.log(x + 1).pow(1 / gamma)
    return x