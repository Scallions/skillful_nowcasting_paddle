import paddle
import netCDF4
import numpy as np
from paddle.io import Dataset, BatchSampler, DataLoader


# PATH = r'E:\dataset\pwv.nc'
# nc_obj = netCDF4.Dataset(PATH)
# print(nc_obj)
# print('---------------------------------------')
#
# #查看nc文件中的变量
# print(nc_obj.variables.keys())
# for i in nc_obj.variables.keys():
#     print(i)
# print('---------------------------------------')
#
# #查看每个变量的信息
# print(nc_obj.variables['time'])
# print(nc_obj.variables['latitude'])
# print(nc_obj.variables['longitude'])
# print(nc_obj.variables['__xarray_dataarray_variable__'])
# print('---------------------------------------')
#
# #查看每个变量的属性
# print(nc_obj.variables['time'].ncattrs())
# print(nc_obj.variables['latitude'].ncattrs())
# print(nc_obj.variables['longitude'].ncattrs())
# print(nc_obj.variables['__xarray_dataarray_variable__'].ncattrs())
# print('---------------------------------------')
#
# #读取数据值
# time=(nc_obj.variables['time'][:])
# lon=(nc_obj.variables['latitude'][:])
# lat=(nc_obj.variables['longitude'][:])
# value = (nc_obj.variables['__xarray_dataarray_variable__'][0])
# print('row',value.shape[0])
# print('col',value.shape[1])
# print(time)
# print(lon)
# print('---------------******-------------------')
# print(lat)
# print('---------------******-------------------')
# print(value.shape)
# print(value)

class NowCastingDataset(Dataset):
    def __init__(self, path, length, ratio, training):
        super().__init__()
        nc_obj = netCDF4.Dataset(path)
        self.value = nc_obj.variables['__xarray_dataarray_variable__']
        self.row = nc_obj.variables['__xarray_dataarray_variable__'].shape[1]
        self.col = nc_obj.variables['__xarray_dataarray_variable__'].shape[2]
        self.total_data = nc_obj.variables['__xarray_dataarray_variable__'].shape[0]
        self.ratio = int(self.total_data * ratio)
        self.length = length
        self.train = training

    def __getitem__(self, item):
        if self.train:
            data = self.value[:self.ratio]
        else:
            data = self.value[self.ratio:]
        inp = data[item:item + self.length]
        tar = data[item + self.length:item + self.length + self.length]
        input = np.resize(np.reshape(inp, (self.length, 1, self.row, self.col)), (self.length, 1, 256, 256))
        target = np.resize(np.reshape(tar, (self.length, 1, self.row, self.col)), (self.length, 1, 256, 256))
        return [input, target]

    def __len__(self):
        if self.train:
            return self.ratio
        else:
            return self.value.shape[0] - self.ratio

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

