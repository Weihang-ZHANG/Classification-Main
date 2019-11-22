import numpy as np
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import os

simple_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

trainset = ImageFolder('./GTSRB/1of8aug', simple_transform)
print("Trainset:", len(trainset))

num_imgs = len(trainset)

means = [0, 0, 0]
stdevs = [0, 0, 0]

for data in trainset:
    img = data[0]
    for i in range(3):
        means[i] += img[i, :, :].mean()
        stdevs[i] += img[i, :, :].std()

means = np.asarray(means) / num_imgs
stdevs = np.asarray(stdevs) / num_imgs

print("normMean = ", means)
print("normStdevs = ", stdevs)

with open("./GTSRB/zscore_1of8aug.txt", 'w') as f1:
    f1.write("normMean: {}, normStdevs: {}".format(means, stdevs))
    f1.write("\n")
    f1.flush()
