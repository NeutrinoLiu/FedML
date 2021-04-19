import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim

import myUtils

EPOCH_NUM = 2
TRAIN_PATH = "GPS-power.dat"
LOSS_PRINT_PER = 500

class FNN(nn.Module):

    # parameter-related operation is defined in init as nn
    def __init__(self):
        super(FNN, self).__init__()
        # input of network is a size2 feature(latitude, longitude)
        self.hl1 = nn.Linear(2,50)  # hiddenlayer 1 
        self.hl2 = nn.Linear(50,50) # hiddenlayer 2
        self.hl3 = nn.Linear(50,50) # hiddenlayer 2
        self.ol = nn.Linear(50,1)   # outputlayer

    # parameter-irrelative operation is recommended as function
    def forward(self, x): # x is the size2 input
        x = F.relu(self.hl1(x))
        x = F.relu(self.hl2(x))
        x = F.relu(self.hl3(x))
        x= self.ol(x)
        return x

class getDataSet(torch.utils.data.Dataset):
    def __init__(self, filePath):
        self.f = open(filePath, "r")
        self.rawlist = self.f.readlines()
        print(f"file {filePath} is read into memory")
        self.f.close()
    
    def __getitem__(self, index):
        row = self.rawlist[index].split(" ")
        lati = myUtils.ux(float(row[0]),1)          # normalization/standardization is a must here, or nothing get learnt 
        longi = myUtils.uy(float(row[1]),1)         # 
        power = float(row[2])
        gps_tensor = torch.tensor([lati, longi])
        power_tensor = torch.tensor([power])
        return gps_tensor, power_tensor

    def __len__(self):
        return len(self.rawlist) 

# dataset init
trainset = getDataSet(TRAIN_PATH) 
trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle = True, num_workers=15) # i7 with 16 cores so use 15 dataloader worker

# training
fnn = FNN()
lossFunc = nn.MSELoss() # min square err

optimizer = optim.SGD(fnn.parameters(), lr = 0.001, momentum = 0) # do not use SGD currently

for epoch in range(EPOCH_NUM):

    avg_loss_per = 0.0 
    for i, data in enumerate(trainloader, 0):
        inputs, truth = data
        optimizer.zero_grad()
        outputs = fnn(inputs)
        loss = lossFunc(outputs, truth)
        loss.backward()
        optimizer.step()

        avg_loss_per += loss.item()
        if i % LOSS_PRINT_PER == LOSS_PRINT_PER-1:
            print(f"[epoch {epoch+1}][avg loss for {LOSS_PRINT_PER} batches before {i+1}] {avg_loss_per/LOSS_PRINT_PER}")
            # print(f"groundTruth is {truth}, prediction is {outputs} ")
            avg_loss_per = 0.0

# for paras in fnn.named_parameters():
#     print(paras)

print("Training done!")
myUtils.visFNN(fnn)