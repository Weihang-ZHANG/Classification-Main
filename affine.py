from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision import utils
import os
import numpy as np

augment_transform = transforms.Compose([
    # transforms.Resize((32,32)),
    transforms.RandomRotation(degrees=30),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor()
])

trainset = ImageFolder('./data/1of4', augment_transform)
print(len(trainset))

trainloader = DataLoader(trainset, batch_size=128, shuffle=False)
minibatch = 0
for data in trainloader:
    minibatch += 1
    img, labels = data
    for i in range(len(labels)):
        if os.path.exists(os.path.join('./data/affine/', str(labels[i].numpy()))) == False:
            os.mkdir(os.path.join('./data/affine/', str(labels[i].numpy())))
        utils.save_image(img[i], os.path.join('./data/affine/', str(labels[i].numpy()), str(minibatch)+'_'+str(i)+'.png'))
        print("Batch OK")

print('Done')
