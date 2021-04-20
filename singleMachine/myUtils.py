import matplotlib.pyplot as plt
import torch
import numpy as np

# range of the whole data set
# lati
xs = 42.987581077
xe = 43.158151564
xavg = 43.07850074790703
xsd = 0.026930841086101193
# longi
ys = -89.564388659
ye = -89.260479204
yavg = -89.3982621182465
ysd = 0.060267757907425355
# power
zs = -86.24192241578515
ze = -27.31659193737493
zavg = -58.52785514280172
zsd = 7.434576197607559

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
        
def uz(raw, type = 0):
    if type == 1:   
        return (raw - zavg) / zsd
    else:
        return (raw - zs) / (ze - zs) * 2 -1

def de_uz(raw, type = 0):
    if type == 1:
        return raw * zsd + zavg
    else:
        return (raw +1)/ 2 * (ze - zs) + zs

def de_proc(loss, type = 0):
    if type == 1:
        return zsd**2 * loss
    else:
        return loss * (ze-zs)**2 / 4

resx = 100
resy = 200

def visFNN(fnn, pp_type = 0): # preprocess mode: 0-norm 1-stand
# range of the visualization (interests)
    xs_in = 42.987581077
    xe_in = 43.158151564
    ys_in = -89.564388659
    ye_in = -89.260479204
    cord_x = np.empty((resx, resy))
    cord_y =np.empty((resx, resy))
    mapvalue = np.empty((resx, resy))
    for i in range(resx):
        xx = xs_in + (xe_in-xs_in)/resx * i
        for j in range(resy):
            yy = ys_in + (ye_in-ys_in)/resy * j
            cord_x[i,j] = xx
            cord_y[i,j] = yy
            inputs = torch.tensor([ux(xx,pp_type), uy(yy,pp_type)])
            outputs = fnn(inputs)
            zz = outputs.item()
            mapvalue[i,j] = de_uz(zz,pp_type)
            # if (i %20 == 0 ) & (j%20 == 0):
            #     print(f"[{format(xx, '.11f')},{format(yy, '.11f')}]:\t{zz}")
    plt.figure(figsize=(9,3))
    plt.pcolormesh(cord_y,cord_x,mapvalue, cmap = "plasma", shading='auto')
    plt.colorbar()
    #plt.show()
    plt.savefig('heatmap.png', bbox_inches='tight')

def visFNN_small(fnn, pp_type = 0):
# range of the visualization (interests)
    xs_in = 43.064
    xe_in = 43.076
    ys_in = -89.45
    ye_in = -89.39
    cord_x = np.empty((resx, resy))
    cord_y =np.empty((resx, resy))
    mapvalue = np.empty((resx, resy))
    for i in range(resx):
        xx = xs_in + (xe_in-xs_in)/resx * i
        for j in range(resy):
            yy = ys_in + (ye_in-ys_in)/resy * j
            cord_x[i,j] = xx
            cord_y[i,j] = yy
            inputs = torch.tensor([ux(xx,pp_type), uy(yy,pp_type)])
            outputs = fnn(inputs)
            zz = outputs.item()
            mapvalue[i,j] = de_uz(zz,pp_type)
            # if (i %20 == 0 ) & (j%20 == 0):
            #     print(f"[{format(xx, '.11f')},{format(yy, '.11f')}]:\t{zz}")
    plt.figure(figsize=(9,3))
    plt.pcolormesh(cord_y,cord_x,mapvalue, cmap = "plasma", shading='auto')
    plt.colorbar()
    #plt.show()
    plt.savefig('heatmap_small.png', bbox_inches='tight')


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
