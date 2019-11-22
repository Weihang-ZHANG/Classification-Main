import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torch.optim as optim
import torch.optim.lr_scheduler
import ResNetModuel
from torchvision import datasets
from torchvision.datasets import ImageFolder

BATCH_SIZE = 128
LR = 0.1


if torch.cuda.is_available():
    print("CUDA inside")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#1of4
normMean1 = [0.44625396, 0.48291734, 0.49262926]
normStd1 = [0.20068598, 0.19916373, 0.20185028]

#1of4aug
normMean2 = [0.45482153, 0.494505, 0.50306225]
normStd2 = [0.21675254, 0.21792613, 0.2192758]

#1of8
normMean3 = [0.44426456, 0.48017612, 0.4903723]
normStd3 = [0.20196006, 0.20022848, 0.20305517]

#1of8aug
normMean4 = [0.44396284, 0.48205075, 0.48990735]
normStd4 = [0.21923937, 0.22048819, 0.22174227]

augment_transform = transforms.Compose([
    transforms.RandomAffine(degrees=30),
    transforms.ColorJitter(contrast=1),
    transforms.ToTensor()
])

singel_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(normMean4, normStd4)
])

trainset = ImageFolder('./data/1of8aug', transform=singel_transform)
testset = ImageFolder('./data/test_cifar10', transform=singel_transform)
print("Trainset:", len(trainset))
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
schedular = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=40, gamma=0.1)

best_acc = 0.0

for epoch in range(n_epochs):

    sum_loss=0.0
    sum_acc=0.0
    test_acc = 0.0

    schedular.step()

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

    if test_acc > best_acc:
        best_acc = test_acc
        best_params = net.state_dict()
        torch.save(best_params, "./data/params_1of8aug_z1.pkl")
        torch.cuda.empty_cache()
        with open("./data/best_1of8aug_z1.txt", 'w') as f1:
            f1.write("Epoch: {}, best_acc: {:.2%}".format(epoch+1, test_acc))
            f1.write("\n")
            f1.flush()

    print("Epoch: {}/{}, Train Acc: {:.2%}, Train Loss: {:.4f}, Test Acc {:.2%}"
         .format(epoch+1, n_epochs, sum_acc, sum_loss, test_acc))
