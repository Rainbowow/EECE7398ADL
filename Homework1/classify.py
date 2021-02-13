# -*- coding:utf-8 -*-
'''
Author: Minghao Chen
Date: 2021-02-08 17:40:52
LastEditTime: 2021-02-13 05:28:02
LastEditors: Please set LastEditors
Description: EECE7398 Homework1: train a classifier
FilePath: \EECE7398ADL\Homework1\Homework1.py
'''

from pickle import FALSE
from utils import *
from Model import *
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import sys

#hyper parameters
PATH = './model/cifar_net.pth'
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
BATCH_SIZE=16

def preprocessing():
    '''
    description: preprocessing the cifar-10 dataset
    param {*}
    return trainloader,testloader
    '''
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=FALSE, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                            shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                            shuffle=False, num_workers=0)

    return trainloader,testloader

trainloader,testloader=preprocessing()

def train():
    
    model=Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    #training
    print_format="{0:<6}{1:<12.4f}{2:<12.4f}{3:<11.4f}{4:<10.4f}"
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        best_loss=10000
        best_acc=0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            #train statisctics
            
            #test statisctics
            
            # print statistics
            
            running_loss += loss.item()
            if i==0:
                print('Loop ','Train Loss ','Train Acc% ','Test Loss ','Test Acc%')
            if i % 2000 == 1999:    # print every 2000 mini-batches

                acc=test(model)
                
                print(print_format.format(str(epoch + 1)+"/10", 1,1,1, running_loss / 2000))
                
                if running_loss<best_loss & acc>best_acc:
                    best_acc=acc
                    torch.save(model.state_dict(), PATH)
                    
                running_loss = 0.0

    print('train finished')
         
def test(model):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))
    return 100 * correct / total

def validate():
    model = Net()
    model.load_state_dict(torch.load(PATH))
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

def main():
    if sys.argv[1]=='train':
        train()
    elif sys.argv[1]=='test':
        validate()
    else:
        print('wrong parameter')

if __name__ == '__main__':
    main()