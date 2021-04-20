import random
import os

FULL_PATH = "GPS-power.dat"
TRAIN_PATH = "spectrum/train/spectrum500_train.json"
TEST_PATH = "spectrum/test/spectrum500_test.json"
TEST_PERCENTAGE = 0.1

if not os.path.exists(os.path.dirname(TRAIN_PATH)):
    os.makedirs(os.path.dirname(TRAIN_PATH))
if not os.path.exists(os.path.dirname(TEST_PATH)):
    os.makedirs(os.path.dirname(TEST_PATH))
fread = open(FULL_PATH, "r")
ftrain = open(TRAIN_PATH, "w+")
ftest = open(TEST_PATH, "w+")

rowlist = fread.readlines()
random.shuffle(rowlist)

test_train_cut = round(TEST_PERCENTAGE * len(rowlist))
for i, row in enumerate(rowlist, 0):
    if i < test_train_cut:
        ftest.write(row)
    else:
        ftrain.write(row)

fread.close()
ftrain.close()
ftest.close()

