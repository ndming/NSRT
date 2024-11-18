import torch
import torch.utils.data as data

class HDF5Dataset(data.Dataset):
    def __init__(self, file):
        super().__init__()
        

def get_dataloader(dataset, config):
    pass


def _tonemap(x, gamma=2.2):
        # See Equation (1) in https://doi.org/10.1145/3543870
        x = torch.clamp(x, 0, 65535)
        x = torch.log(x + 1).pow(1 / gamma)
        return x