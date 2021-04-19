import numpy as np

FILE_NAME = "shrinked0-1572bins.dat"
OUTPUT = "GPS-power.dat"

def cal_total_power(readings_dB, start, end):
    total_power = np.sum(10.0 ** (readings_dB[start:end]/10.0))
    total_power_dB = 10. * np.log10(total_power)
    return total_power_dB

f = open(FILE_NAME, "r")
fw = open(OUTPUT, "w+")

xmin = 100.0
xmax = 0.0
ymin = 0.0
ymax = -100.0
zmin = 0.0
zmax = -100.0
xavg = 0.0
yavg = 0.0
zavg = 0.0

lines = f.readlines()
for row in lines:
    dataList = row.split(" ")
    power = cal_total_power(np.array(list(map(float, dataList))), 2, len(dataList))
    lati = dataList[0]
    xmin = min(xmin, float(lati))
    xmax = max(xmax, float(lati))
    xavg += float(lati)
    longi = dataList[1]
    ymin = min(ymin, float(longi))
    ymax = max(ymax, float(longi))
    yavg += float(longi)
    fw.write(lati + ' ' + longi + ' ' + str(power) + '\n')
    zmin = min(zmin, power)
    zmax = max(zmax, power)
    zavg += power

xavg = xavg/len(lines)
yavg = yavg/len(lines)
zavg = zavg/len(lines)
xsd = 0.0
ysd = 0.0
zsd = 0.0
for row in lines:
    dataList = row.split(" ")
    lati = float(dataList[0])
    longi = float(dataList[1])
    power = cal_total_power(np.array(list(map(float, dataList))), 2, len(dataList))
    xsd += (lati-xavg)**2
    ysd += (longi-yavg)**2
    zsd += (power-zavg)**2

xsd = (xsd/len(lines))**0.5
ysd = (ysd/len(lines))**0.5
zsd = (zsd/len(lines))**0.5

print(f"lati:{xmin}~{xmax} [avg:{xavg}, sd:{xsd}], longi:{ymin}~{ymax} [avg:{yavg}, sd:{ysd}], power:{zmin}~{zmax} [avg:{zavg}, sd:{zsd}]")
fw.close()
f.close()


