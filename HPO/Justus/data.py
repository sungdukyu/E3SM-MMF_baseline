import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

"""
Contains Methods for Data Preprocessing and Loading.
Droplet distribution simulation data provided by Jerry Lin, available on PSC's Bridges-2.

E3SM-MMF.mlo.{%4d}-{%2d}-{%2d}-{%5d}.nc - [bins(33), time(1), X(640), Y(640), Z(75)]
    - mixing ratio, netCDF4, ~196GB (49 * ~4B) (* 6+ metrological cases, e.g. 'atex', 'dycoms')

train_input.npy / train_target.npy [n(10.091.520), vars(124)] -> [..., vars(128)]
val_input.npy / val_target.npy [n(1.441.920), vars(124)] -> [..., vars(128)]
"""

DATA_PATH_NPY = 'data/'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Save loaded data globally
datasets = [None, None]


class NumpyData(Dataset):

    def __init__(self, dir_x, dir_y, process_x=None, process_y=None, transform=None):
        self.transform = transform
        self._norm = None

        self.datax = torch.from_numpy(np.load(dir_x))
        if process_x is not None:
            self.datax = process_x(self.datax)
        self.datay = torch.from_numpy(np.load(dir_y))
        if process_y is not None:
            self.datay = process_y(self.datay)

    def __len__(self):
        return self.datax.shape[0]

    @property
    def norm(self):
        if self._norm is None:
            # self._norm = [np.ones_like(self.datax.shape[0]), np.ones_like(self.datax.shape[1])]
            self._norm = [self.datax.std(axis=0), self.datay.std(axis=0)]
        return self._norm

    def __getitem__(self, idx):
        x = self.datax[idx]
        y = self.datay[idx]
        if self.transform is not None:
            mx, my = self.norm[0] != 0, self.norm[1] != 0
            x[:, mx] = x[:, mx] / self.norm[0][mx]
            y[:, my] = y[:, my] / self.norm[1][my]
            # x = self.transform(x)
        return {'x': x, 'y': y}

    def __getitems__(self, idxs):
        return self[idxs]

    def dataloader(self, **kwargs):
        return DataLoader(self, collate_fn=lambda x: x, **kwargs)


def get_data(val_only=False, **kwargs):

    global datasets
    to_gpu = lambda x: x
    # to_gpu = lambda x: x.to(device)
    if datasets[0] is None and not val_only:
        datasets = [NumpyData(*[DATA_PATH_NPY + '%s_%s.npy' % (t, s) for s in ['input', 'target']],
                              process_x=to_gpu, process_y=to_gpu) for t in ['train', 'val']]
    elif datasets[1] is None and val_only:
        datasets[1] = NumpyData(*[DATA_PATH_NPY + 'val_%s_stride6.npy' % s for s in ['input', 'target']],
                                process_x=to_gpu, process_y=to_gpu)
    return [d.dataloader(**kwargs) if d is not None else None for d in datasets]


if __name__ == "__main__":
    # Load validation data
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
