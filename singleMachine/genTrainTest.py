import random
import os
import json
import math
import myUtils 
import sys

FULL_PATH = "GPS-power.dat"
TRAIN_PATH = "spectrum/train/spectrum500_train.json"
TEST_PATH = "spectrum/test/spectrum500_test.json"
TEST_PERCENTAGE = 0.1

if __name__ == "__main__":
    if len(sys.argv) > 1 :
        if int(sys.argv[1]) > 1 :
            CLIENT_NUM = int(sys.argv[1])
    else:
        CLIENT_NUM = 10 # default is 10 worker
        # NOTICE: now you DO NOT need to change CLIENT_NUM in this file, it is directly passed by shell argument

print(f"[DataSet Generator]\tthere will be {CLIENT_NUM} client devices")

def genJson(dataSet,targetPath):
    if not os.path.exists(os.path.dirname(targetPath)):
        os.makedirs(os.path.dirname(targetPath))
    f = open(targetPath, "w+")

    users = [ "uid_"+str(i) for i in range(CLIENT_NUM)]
    size = math.ceil(len(dataSet) / CLIENT_NUM)
    xandy = [dataSet[i:i+size] for i in range(0, len(dataSet), size)]

    all_x = []
    all_y = []
    num = []
    for subset in xandy:
        num += [len(subset)]
        local_x = []
        local_y = []
        for row in subset:
            local_x += [[myUtils.ux(float(row.split(' ')[0])), \
                         myUtils.uy(float(row.split(' ')[1]))]]
            local_y += [[myUtils.uz(float(row.split(' ')[2]))]]
        all_x += [local_x]
        all_y += [local_y]
    
    outputJson = {}
    outputJson["num_samples"] = num
    outputJson["users"] = users
    
    allUserdata = {}
    for i in range(CLIENT_NUM):
        userJson = {}
        userJson["y"] = all_y[i]
        userJson["x"] = all_x[i]
        allUserdata[users[i]] = userJson

    outputJson["user_data"] = allUserdata
    f.write(json.dumps(outputJson, indent = 4))

    f.close()

fread = open(FULL_PATH, "r")
fullList = fread.readlines()
fread.close()

random.seed()
random.shuffle(fullList)

cutter = round(TEST_PERCENTAGE * len(fullList))
testList = fullList[0:cutter]
trainList = fullList[cutter:len(fullList)]

genJson(testList, TEST_PATH)
genJson(trainList, TRAIN_PATH)


