import matplotlib.pyplot as plt
import torch
import numpy as np

xs = 42.987581077
xe = 43.158151564
ys = -89.564388659
ye = -89.260479204

# xs = 43.064
# xe = 43.076
# ys = -89.45
# ye = -89.39

resx = 100
resy = 200
mapvalue = np.empty((resx, resy))

def visFNN(fnn):
    for i in range(resx):
        for j in range(resy):
            xx = xs + (xe-xs)/resx * i
            yy = ys + (ye-ys)/resy * j
            inputs = torch.tensor([xx, yy])
            outputs = fnn(inputs)
            zz = outputs.item()
            mapvalue[i,j] = zz 
            if (i %20 == 0 ) & (j%20 == 0):
                print(f"[{format(xx, '.11f')},{format(yy, '.11f')}]:\t{zz}")
    plt.pcolormesh(mapvalue, cmap = "plasma")
    #plt.pcolormesh(mapvalue, cmap = "summer")
    plt.show()

# FILE_NAME = "GPS-power.dat"
# f = open(FILE_NAME, "r")
# x = []  #lati
# y = []  #longi
# v = []  #power

# for rows in f.readlines():
#     xyv = rows.split(" ")
#     x.append(float(xyv[0]))
#     y.append(float(xyv[1]))
#     v.append(float(xyv[2]))

# x_floor = min(x)
# x_ceil = max(x)
# y_floor = min(y)
# y_ceil = max(y)

# fig = plt.figure()
# draw = fig.add_subplot(projection = '3d')

# for xx, yy, zz in zip(x,y,v):
#     draw.scatter(xx,yy,zz)

# plt.show()
# f.close()
