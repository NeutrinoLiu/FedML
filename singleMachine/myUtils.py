import matplotlib.pyplot as plt
import torch
import numpy as np

# range of the whole data set
xs = 42.987581077
xe = 43.158151564
xavg = 43.07850074790703
xsd = 0.026930841086101193
ys = -89.564388659
ye = -89.260479204
yavg = -89.3982621182465
ysd = 0.060267757907425355


# range of the visualization (interests)
xs_in = 42.987581077
xe_in = 43.158151564
ys_in = -89.564388659
ye_in = -89.260479204

# xs_in = 43.064
# xe_in = 43.076
# ys_in = -89.45
# ye_in = -89.39


def ux(raw, type = 0):
    if type == 1:   
        return (raw - xavg) / xsd
    else:
        return (raw - xs) / (xe - xs) * 2 -1

def uy(raw, type = 0):
    if type == 1:   
        return (raw - yavg) / ysd
    else:
        return (raw - ys) / (ye - ys) * 2 -1

resx = 100
resy = 200
mapvalue = np.empty((resx, resy))

def visFNN(fnn):
    for i in range(resx):
        for j in range(resy):
            xx = xs_in + (xe_in-xs_in)/resx * i
            yy = ys_in + (ye_in-ys_in)/resy * j
            inputs = torch.tensor([ux(xx), uy(yy)])
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
