# Code adapted from:
# https://github.com/chengtan9907/OpenSTL

import warnings
warnings.filterwarnings("ignore")

import random
import numpy as np
import os.path as osp
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from lib.datasets.utils import create_loader

try:
    import xarray as xr
except ImportError:
    xr = None

d2r = np.pi / 180


def latlon2xyz(lat, lon):
    if type(lat) == torch.Tensor:
        x = -torch.cos(lat)*torch.cos(lon)
        y = -torch.cos(lat)*torch.sin(lon)
        z = torch.sin(lat)

    if type(lat) == np.ndarray:
        x = -np.cos(lat)*np.cos(lon)
        y = -np.cos(lat)*np.sin(lon)
        z = np.sin(lat)
    return x, y, z


def xyz2latlon(x, y, z):
    if type(x) == torch.Tensor:
        lat = torch.arcsin(z)
        lon = torch.atan2(-y, -x)

    if type(x) == np.ndarray:
        lat = np.arcsin(z)
        lon = np.arctan2(-y, -x)
    return lat, lon


data_map = {
    'ocean_t0': 'water_temp_depth_0m',
    'ocean_s0': 'salinity_depth_0m',
    'ocean_u0': 'water_u_depth_0m',
    'ocean_v0': 'water_v_depth_0m',
    'ocean_t': 'water_temp',
    'ocean_s': 'salinity',
    'ocean_u': 'water_u',
    'ocean_v': 'water_v'
}

mv_data_map = {
    **dict.fromkeys(['mv', 'mv4'], ['ocean_t', 'ocean_s', 'ocean_u', 'ocean_v']),
    'mv5': ['g', 'r', 't', 'u', 'v'],
}


class OceanHYCOMDataset(Dataset):
    """Ocean HYCOM Dataset `_

    Args:
        data_root (str): Path to the dataset.
        data_name (str): Name of the Ocean modality in HYCOM.
        training_time (list): The arrange of years for training.
        idx_in (list): The list of input indices.
        idx_out (list): The list of output indices to predict.
        step (int): Sampling step in the time dimension.
        level (int): Used level in the multi-variant version.
        data_split (str): The resolution (degree) of Ocean Bench splits.
        use_augment (bool): Whether to use augmentations (defaults to False).
    """

    def __init__(self, data_root, data_name, training_time,
                 idx_in, idx_out, temp_stride=1, step=1, level=1, data_split='32_64',
                 mean=None, std=None,
                 transform_data=None, transform_labels=None, use_augment=False):
        super().__init__()
        self.data_root = data_root
        self.data_name = data_name
        self.data_split = data_split
        self.training_time = training_time
        self.idx_in = np.array(idx_in) * temp_stride
        self.idx_out = np.array(idx_out) * temp_stride
        self.step = step
        self.level = level
        self.data = None
        self.mean = mean
        self.std = std
        self.transform_data = transform_data
        self.transform_labels = transform_labels
        self.use_augment = use_augment
        assert isinstance(level, (int, list))

        self.time = None
        shape = data_split.split("_")
        self.shape = (int(shape[0]), int(shape[1]))

        if isinstance(data_name, list):
            data_name = data_name[0]
        if 'mv' in data_name:  # multi-variant version
            self.data_name = mv_data_map[data_name]
            self.data, self.mean, self.std = [], [], []
            for name in self.data_name:
                data, mean, std = self._load_data_xarray(data_name=name, single_variant=False)
                self.data.append(data)
                self.mean.append(mean)
                self.std.append(std)
            self.data = np.concatenate(self.data, axis=1)
            self.mean = np.concatenate(self.mean, axis=1)
            self.std = np.concatenate(self.std, axis=1)
        else:  # single variant
            self.data_name = data_name
            self.data, mean, std = self._load_data_xarray(data_name, single_variant=True)
            if self.mean is None:
                self.mean, self.std = mean, std

        self.valid_idx = np.array(
            range(-self.idx_in[0], self.data.shape[0]-self.idx_out[-1]-1))

    def _load_data_xarray(self, data_name, single_variant=True):
        """Loading full data with xarray"""
        if data_name != 'ocean_uv0':
            try:
                dataset = xr.open_mfdataset(self.data_root+'/{}/{}*.nc'.format(
                    data_map[data_name], data_map[data_name]), combine='by_coords')
            except (AttributeError, ValueError):
                assert False and 'Please install xarray and its dependency (e.g., netcdf4), ' \
                                    'pip install xarray==0.19.0,' \
                                    'pip install netcdf4 h5netcdf dask'
            except OSError:
                print("OSError: Invalid path {}/{}/*.nc".format(self.data_root, data_map[data_name]))
                assert False
            dataset = dataset.sel(time=slice(*self.training_time))
            dataset = dataset.isel(time=slice(None, -1, self.step))
            if self.time is None and single_variant:
                self.week = dataset['time.week']
                self.month = dataset['time.month']
                self.year = dataset['time.year']
                self.time = np.stack(
                    [self.week, self.month, self.year], axis=1)
                lon, lat = np.meshgrid(
                    dataset.lon*d2r, dataset.lat*d2r)
                x, y, z = latlon2xyz(lat, lon)
                self.V = np.stack([x, y, z]).reshape(3, self.shape[0]*self.shape[1]).T
            if not single_variant and isinstance(self.level, list):
                dataset = dataset.sel(depth=np.array(self.level))
            data = dataset.get(data_name.split("_")[1]).values[:, np.newaxis, :, :]

        elif data_name == 'ocean_uv0':
            input_datasets = []
            for key in ['ocean_u0', 'ocean_v0']:
                try:
                    dataset = xr.open_mfdataset(self.data_root+'/{}/{}*.nc'.format(
                        data_map[key], data_map[key]), combine='by_coords')
                except (AttributeError, ValueError):
                    assert False and 'Please install xarray and its dependency (e.g., netcdf4), ' \
                                     'pip install xarray==0.19.0,' \
                                     'pip install netcdf4 h5netcdf dask'
                except OSError:
                    print("OSError: Invalid path {}/{}/*.nc".format(self.data_root, data_map[key]))
                    assert False
                dataset = dataset.sel(time=slice(*self.training_time))
                dataset = dataset.isel(time=slice(None, -1, self.step))
                if self.time is None and single_variant:
                    self.week = dataset['time.week']
                    self.month = dataset['time.month']
                    self.year = dataset['time.year']
                    self.time = np.stack(
                        [self.week, self.month, self.year], axis=1)
                    lon, lat = np.meshgrid(
                        dataset.lon*d2r, dataset.lat*d2r)
                    x, y, z = latlon2xyz(lat, lon)
                    self.V = np.stack([x, y, z]).reshape(3, self.shape[0]*self.shape[1]).T
                input_datasets.append(dataset.get(key.split("_")[1]).values[:, np.newaxis, :, :])
            data = np.concatenate(input_datasets, axis=1)

        # uv0
        if len(data.shape) == 5:
            data = data.squeeze(1)
        # multi-variant level
        if not single_variant and isinstance(self.level, int):
            data = data[:, -self.level:, ...]

        # mean_raw = data.mean(axis=(0, 2, 3)).reshape(1, data.shape[1], 1, 1)
        # var_raw = data.var(axis=(0, 2, 3)).reshape(1, data.shape[1], 1, 1)
        # zero_raw = np.zeros(mean_raw.shape, dtype=np.float32)
        # mean = (mean_raw * 2048) / (2048 - 644)
        # var = (var_raw * 2048 - np.square(zero_raw - mean_raw) * 644) / (2048 - 644)
        # std = np.sqrt(var)

        mean = data.mean(axis=(0, 2, 3)).reshape(1, data.shape[1], 1, 1)
        std = data.std(axis=(0, 2, 3)).reshape(1, data.shape[1], 1, 1)
        # mean = dataset.mean('time').mean(('lat', 'lon')).compute()[data_name].values
        # std = dataset.std('time').mean(('lat', 'lon')).compute()[data_name].values
        data = (data - mean) / std

        return data, mean, std

    def _augment_seq(self, seqs, crop_scale=0.96):
        """Augmentations as a video sequence"""
        _, _, h, w = seqs.shape  # original shape, e.g., [4, 1, 128, 256]
        seqs = F.interpolate(seqs, scale_factor=1 / crop_scale, mode='bilinear')
        _, _, ih, iw = seqs.shape
        # Random Crop
        x = np.random.randint(0, ih - h + 1)
        y = np.random.randint(0, iw - w + 1)
        seqs = seqs[:, :, x:x+h, y:y+w]
        # Random Flip
        if random.randint(0, 1):
            seqs = torch.flip(seqs, dims=(3, ))  # horizontal flip
        return seqs

    def __len__(self):
        return self.valid_idx.shape[0]

    def __getitem__(self, index):
        index = self.valid_idx[index]
        data = torch.tensor(self.data[index+self.idx_in])
        labels = torch.tensor(self.data[index+self.idx_out])
        if self.use_augment:
            len_data = self.idx_in.shape[0]
            seqs = self._augment_seq(torch.cat([data, labels], dim=0), crop_scale=0.96)
            data = seqs[:len_data, ...]
            labels = seqs[len_data:, ...]
        return data, labels


def load_data(batch_size,
              val_batch_size,
              data_root,
              num_workers=4,
              data_split='32_64',
              data_name='t0',
              train_time=['1994', '2013'],
              val_time=['2014', '2014'],
              test_time=['2015', '2015'],
              idx_in=[i for i in range(-15, 1)],
              idx_out=[i for i in range(1, 17)],
              step=1,
              level=1,
              distributed=False, use_augment=False, use_prefetcher=False, drop_last=False,
              **kwargs):

    assert data_split in ['32_64', '64_128']
    _dataroot = osp.join(data_root, f'ocean_{data_split}')
    ocean_dataroot = _dataroot if osp.exists(_dataroot) else osp.join(data_root, 'ocean')

    train_set = OceanHYCOMDataset(data_root=ocean_dataroot,
                                    data_name=data_name, data_split=data_split,
                                    training_time=train_time,
                                    idx_in=idx_in,
                                    idx_out=idx_out,
                                    temp_stride=kwargs.get('temp_stride', 1),
                                    step=step, level=level, use_augment=use_augment)
    vali_set = OceanHYCOMDataset(ocean_dataroot,
                                    data_name=data_name, data_split=data_split,
                                    training_time=val_time,
                                    idx_in=idx_in,
                                    idx_out=idx_out,
                                    temp_stride=kwargs.get('temp_stride', 1),
                                    step=step, level=level, use_augment=False,
                                    mean=train_set.mean,
                                    std=train_set.std)
    test_set = OceanHYCOMDataset(ocean_dataroot,
                                    data_name, data_split=data_split,
                                    training_time=test_time,
                                    idx_in=idx_in,
                                    idx_out=idx_out,
                                    temp_stride=kwargs.get('temp_stride', 1),
                                    step=step, level=level, use_augment=False,
                                    mean=train_set.mean,
                                    std=train_set.std)

    dataloader_train = create_loader(train_set,
                                     batch_size=batch_size,
                                     shuffle=True, is_training=True,
                                     pin_memory=True, drop_last=True,
                                     num_workers=num_workers,
                                     distributed=distributed, use_prefetcher=use_prefetcher)
    dataloader_vali = create_loader(test_set, # validation_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher)
    dataloader_test = create_loader(test_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher)

    return dataloader_train, dataloader_vali, dataloader_test


if __name__ == '__main__':
    print()
