import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision import models
import torch.optim.lr_scheduler
import ResNetModuel
import matplotlib.pyplot as plt
import numpy as np


BATCH_SIZE = 128
LR = 0.1


if torch.cuda.is_available():
    print("CUDA inside")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#1of4
normMean1 = [0.34085074, 0.312759, 0.3219871]
normStd1 = [0.1606724, 0.15970361, 0.16878754]

#1of4aug
normMean2 = [0.39966425, 0.36888418, 0.383248]
normStd2 = [0.20040484, 0.2001912, 0.2108042]

#1of8
normMean3 = [0.3448876,  0.31641838, 0.3252931]
normStd3 = [0.16191217, 0.16100039, 0.16972816]

#1of8aug
normMean4 = [0.38759044, 0.35849798, 0.37162146]
normStd4 = [0.20522162, 0.20592165, 0.2164248]

#original
normMean5 = [0.34025675, 0.3121438,  0.321437]
normStd5 = [0.15954028, 0.15901476, 0.16826406]

#4times
normMean6 = [0.39980784, 0.36903089, 0.38352707]
normStd6 = [0.19958616, 0.1998527,  0.21054134]

#enhancedTrain
normMean7 = [0.39970428, 0.37020406, 0.38515702]
normStd7 = [0.20783012, 0.21040893, 0.2206835]

#enhanced
normMean8 = [0.39648438, 0.36850678, 0.3825123]
normStd8 = [0.21178492, 0.21430861, 0.22266305]

#affine
normMean9 = [0.28185454, 0.25570723, 0.26529416]
normStd9 = [0.19814205, 0.18949243, 0.19920449]

#1of4hm
normMean10 = [0.47071236, 0.46248665, 0.4650983]
normStd10 = [0.12145051, 0.1204391,  0.12229938]

#hmTrain
normMean11 = [0.47102193, 0.46264216, 0.4652844]
normStd11 = [0.12196087, 0.12098021, 0.12292381]

#1of8hm
normMean12 = [0.49461758, 0.4895518,  0.491026]
normStd12 = [0.1139341,  0.11290205, 0.11363581]

#histeq
normMean13 = [0.53495926, 0.4726983, 0.50758445]
normStd13 = [0.27758428, 0.29285803, 0.2932313]

simple_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(normMean5, normStd5)
])

augment_transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.RandomRotation(degrees=30),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(normMean5, normStd5)
])

trainset = ImageFolder('./GTSRB/Train', augment_transform)
print("Trainset:", len(trainset))
testset = ImageFolder('./GTSRB/Test', simple_transform)
print("Testset:", len(testset))


trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=3)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=3)


net = ResNetModuel.ResNet18()
# ResNetModuel.test()



if torch.cuda.is_available():
    net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)
n_epochs = 160
schedular = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

best_acc = 0.0

# train_list = []
# test_list = []

for epoch in range(n_epochs):

    sum_loss=0.0
    sum_acc=0.0
    test_acc = 0.0



    net.train()
    for i, data in enumerate(trainloader, 0):


        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        # forward & backward
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()


        sum_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum()
        sum_acc += correct.item()



    net.eval()
    for data in testloader:

        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).sum()
        test_acc += correct.item()

    test_acc /= len(testset)
    sum_loss /= len(trainset)
    sum_acc /= len(trainset)

    # train_list.append(sum_acc)
    # test_list.append(test_acc)

    if test_acc > best_acc:
        best_acc = test_acc
        best_params = net.state_dict()
        torch.save(best_params, "./GTSRB/pkl/orig_affine_new_1.pkl")
        torch.cuda.empty_cache()
        with open("./GTSRB/acc/orig_affine_new_1.txt", 'w') as f1:
            f1.write("Epoch: {}, best_acc: {:.2%}".format(epoch+1, test_acc))
            f1.write("\n")
            f1.flush()

    schedular.step()
    current_lr = schedular.get_lr()

    print("Epoch: {}/{}, Train Acc: {:.2%}, Train Loss: {:.4f}, Test Acc: {:.2%}, lr: {}"
         .format(epoch+1, n_epochs, sum_acc, sum_loss, test_acc, current_lr))


# x1 = range(160)
# x2 = range(160)
# y1 = train_list
# y2 = test_list
# plt.subplot(2,1,1)
# plt.plot(x1,y1,'-')
# np.save('histeq.npy', train_list)
# np.save('1of8augtest.npy', test_list)
