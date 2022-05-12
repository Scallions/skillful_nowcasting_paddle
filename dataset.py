import paddle
import xarray as xr
import numpy as np
from paddle.io import Dataset, BatchSampler, DataLoader

class NowCastingDataset(Dataset):
    def __init__(self, path, length, ratio, training):
        super().__init__()
        self.value = xr.load_dataarray(path)
        self.lats = np.linspace(self.value.latitude[0], self.value.latitude[-1], 256)
        self.lons = np.linspace(self.value.longitude[0], self.value.longitude[-1], 256)
        self.total_data = self.value.shape[0]
        self.ratio = int(self.total_data * ratio)
        self.length = length
        self.length_target = 4
        self.train = training

    def __getitem__(self, item):
        if self.train:
            data = self.value[:self.ratio]
        else:
            data = self.value[self.ratio:]
        inp = data[item:item + self.length]
        tar = data[item + self.length:item + self.length + self.length_target]
        # input = np.resize(np.reshape(inp, (self.length, 1, self.row, self.col)), (self.length, 1, 256, 256))
        # target = np.resize(np.reshape(tar, (self.length_target, 1, self.row, self.col)), (self.length_target, 1, 256, 256))
        input = inp.interp(longitude=self.lons, latitude=self.lats).values
        target = tar.interp(longitude=self.lons, latitude=self.lats).values
        input = np.reshape(input, (self.length, 1, 256, 256))
        target = np.reshape(target, (self.length_target, 1, 256, 256))
        return [input, target]

    def __len__(self):
        if self.train:
            return self.ratio
        else:
            return self.value.shape[0] - self.ratio - 4

if __name__ == '__main__':
    PATH = r'E:\dataset\pwv.nc'
    LENGTH = 22
    dataset = NowCastingDataset(PATH, LENGTH, 0.8, training=False)
    loader = DataLoader(dataset,
                        batch_size=64,
                        shuffle=False,
                        drop_last=True,
                        num_workers=0)
    for i, (inp, target) in enumerate(loader):
        print(i)
        print(inp.shape)
        print(target.shape)
        break
    print('fininshed!')

