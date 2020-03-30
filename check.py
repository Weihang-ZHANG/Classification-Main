import torch.utils.data
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision import utils
import torch.optim.lr_scheduler
import ResNetModuel
import os.path
import numpy as np


BATCH_SIZE = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#1of4
normMean1 = [0.34085074, 0.312759, 0.3219871]
normStd1 = [0.1606724, 0.15970361, 0.16878754]

#1of4aug
normMean2 = [0.39966425, 0.36888418, 0.383248]
normStd2 = [0.20040484, 0.2001912, 0.2108042]

#1of8
normMean3 = [0.3457714, 0.3171815, 0.3261122]
normStd3 = [0.162222, 0.161295, 0.16999497]

#1of8aug
normMean4 = [0.38801074, 0.35879454, 0.37196222]
normStd4 = [0.20525436, 0.20591636, 0.21635535]

#original
normMean5 = [0.34025675, 0.3121438,  0.321437]
normStd5 = [0.15954028, 0.15901476, 0.16826406]

#4times
normMean6 = [0.39980784, 0.36903089, 0.38352707]
normStd6 = [0.19958616, 0.1998527,  0.21054134]

#1of4hm
normMean10 = [0.47071236, 0.46248665, 0.4650983]
normStd10 = [0.12145051, 0.1204391,  0.12229938]

def saveimg(inp):
    inp = inp.numpy().transpose((1,2,0))
    mean = np.array(normMean1)
    std = np.array(normStd1)
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = transform(inp)
    return inp

simple_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(normMean10, normStd10)
])

transform = transforms.Compose([
    transforms.ToTensor()
])

testset = ImageFolder('./GTSRB/recombination/low', transform=simple_transform)
print("Testset:", len(testset))

testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=3)


net = ResNetModuel.ResNet18()

if torch.cuda.is_available():
    net.cuda()

net.eval()
net.load_state_dict(torch.load('/home/ZWH1573/PycharmProjects/ResNet/GTSRB/1of4_hm_3.pkl', map_location='cpu'))

test_acc = 0.0

for data in testloader:

    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)

    outputs = net(inputs)
    _, predicted = torch.max(outputs, 1)

    result = predicted == labels


    correct = (predicted == labels).sum()
    test_acc += correct.item()

test_acc /= len(testset)
print('Test accuracy:{:.2%}' .format(test_acc))
