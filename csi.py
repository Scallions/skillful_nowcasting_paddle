from dataset import NowCastingDataset
from models.generators import Generator
import paddle


G = Generator(num_channels=1, lead_time=90, time_delta=5)
# G.eval()
para_state_dict = paddle.load("/Volumes/HDD/Data/paddle/G_11000_model.pdparams")
G.set_state_dict(para_state_dict)


PATH = "/Volumes/HDD/Data/ztd/pwv.nc"
LENGTH = 18
BATCH_SIZE = 1

test_dataset = NowCastingDataset(PATH, LENGTH, 0.8, training=False)


inp, target = test_dataset[8000]
inp = paddle.to_tensor(inp).unsqueeze(0).astype(paddle.float32)
target = paddle.to_tensor(target).unsqueeze(0).astype(paddle.float32)
print(inp.shape, target.shape)

pred = G(inp)

print(pred.shape)

# to numpy
inp = inp.numpy()
out = pred.numpy()
target = target.numpy()

import tools.scores as scores
import matplotlib.pyplot as plt
from cartopy.crs import LambertAzimuthalEqualArea, PlateCarree
import matplotlib as mpl
import numpy as np

di = 3
dt = target[0,di,0,::-1,:] #- target[0,0,0,::-1,:]
dp = out[0,di,0,::-1,:] #- out[0,0,0,::-1,:]
# plt.imshow(dp)
# plt.show()
csi = scores.TS(dt, dp, 1)
print(f"CSI: {csi}")


proj = LambertAzimuthalEqualArea(-40, 60)
cmap = mpl.colormaps["jet"]
cdata = "2019-2-1"
vmin = 0
vmax = 20

fig, axs = plt.subplots(1,2, subplot_kw={'projection': PlateCarree()})
ax = axs[0]
ax.set_title(f"PRED:{cdata}")
ax.set_extent([-75,-12,55,85])
ax.coastlines()
ax.gridlines(draw_labels=True)
x = np.linspace(-75, -12, 256)
y = np.linspace(55, 85, 256)
xs, ys = np.meshgrid(x,y)
mesh = ax.pcolormesh(xs, ys, out[0,di,0,::-1,:], vmin=vmin, vmax= vmax, transform=PlateCarree(), cmap = cmap)
# plt.imshow(out[pidx,:,:])
# plt.axis('off')
ax = axs[1]
ax.set_title(f"ERA:{cdata}")
ax.set_extent([-75,-12,55,85])
ax.coastlines()
ax.gridlines(draw_labels=True)

mesh = ax.pcolormesh(xs, ys, target[0,di,0,::-1,:], vmin=vmin, vmax=vmax, transform=PlateCarree(), cmap=cmap)
# plt.colorbar(ax=ax)
fig.colorbar(mesh, ax=axs, orientation='horizontal')

plt.show()
