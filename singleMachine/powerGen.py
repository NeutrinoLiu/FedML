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

for row in f.readlines():
    dataList = row.split(" ")
    power = cal_total_power(np.array(list(map(float, dataList))), 2, len(dataList))
    lati = dataList[0]
    xmin = min(xmin, float(lati))
    xmax = max(xmax, float(lati))
    longi = dataList[1]
    ymin = min(ymin, float(longi))
    ymax = max(ymax, float(longi))
    fw.write(lati + ' ' + longi + ' ' + str(power) + '\n')
    zmin = min(zmin, power)
    zmax = max(zmax, power)

print(f"lati:{xmin}~{xmax}, longi:{ymin}~{ymax}, power:{zmin}~{zmax}")
fw.close()
f.close()


