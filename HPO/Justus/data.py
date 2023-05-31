import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

"""
Contains Methods for Data Preprocessing and Loading.
Droplet distribution simulation data provided by Jerry Lin, available on PSC's Bridges-2.

E3SM-MMF.mlo.{%4d}-{%2d}-{%2d}-{%5d}.nc - [bins(33), time(1), X(640), Y(640), Z(75)]
    - mixing ratio, netCDF4, ~196GB (49 * ~4B) (* 6+ metrological cases, e.g. 'atex', 'dycoms')

train_input.npy / train_target.npy [n(10.091.520), vars(128)] -> [..., vars(128)]
val_input.npy / val_target.npy [n(1.441.920), vars(128)] -> [..., vars(128)]
"""

DATA_PATH_NPY = 'Data/'


class NumpyData(Dataset):

    def __init__(self, dir_x, dir_y, process_x=None, process_y=None, transform=None):
        self.transform = transform

        self.datax = torch.from_numpy(np.load(dir_x))
        if process_x is not None:
            self.datax = process_x(self.datax)
        self.datay = torch.from_numpy(np.load(dir_y))
        if process_y is not None:
            self.datay = process_y(self.datax)

    def __len__(self):
        return self.datax.shape[0]

    def __getitem__(self, idx):
        x = self.datax[idx]
        y = self.datay[idx]
        if self.transform is not None:
            x = self.transform(x)
        return {'x': x, 'y': y}

    def __getitems__(self, idxs):
        return self[idxs]

    def dataloader(self, **kwargs):
        return DataLoader(self, collate_fn=lambda x: x, **kwargs)


if __name__ == "__main__":
    # Load validation
    import time
    data = NumpyData(DATA_PATH_NPY + 'val_input.npy', DATA_PATH_NPY + 'val_target.npy').dataloader(batch_size=4096)
    batch = next(iter(data))
    print(batch['x'].shape)
    print(batch['y'].shape)

    # Time loading for full epoch
    s = time.time()
    for b in data:
        continue
    print(time.time() - s)
