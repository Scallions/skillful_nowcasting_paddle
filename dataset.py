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
    def __init__(self, path, length):
        super().__init__()
        nc_obj = netCDF4.Dataset(path)
        self.value = nc_obj.variables['__xarray_dataarray_variable__']
        self.row = nc_obj.variables['__xarray_dataarray_variable__'].shape[1]
        self.col = nc_obj.variables['__xarray_dataarray_variable__'].shape[2]
        self.length = length

    def __getitem__(self, item):
        inp = self.value[item:item + self.length]
        target = self.value[item + self.length:item + self.length + self.length]
        inp = np.reshape(inp, (self.length, 1, self.row, self.col))
        target = np.reshape(target, (self.length, 1, self.row, self.col))
        return [inp, target]

    def __len__(self):
        return self.value.shape[0]

if __name__ == '__main__':
    PATH = r'E:\dataset\pwv.nc'
    LENGTH = 22
    dataset = NowCastingDataset(PATH,LENGTH)
    loader = DataLoader(dataset,
                        batch_size=4,
                        shuffle=True,
                        drop_last=True,
                        num_workers=0)
    for i, (inp, target) in enumerate(loader):
        print(i)
        print(inp.shape)
        print(target.shape)
        break

