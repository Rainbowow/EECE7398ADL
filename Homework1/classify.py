# -*- coding:utf-8 -*-
'''
Author: Minghao Chen
Date: 2021-02-08 17:40:52
LastEditTime: 2021-02-17 23:42:59
LastEditors: Please set LastEditors
Description: EECE7398 Homework1: train a classifier
FilePath: \EECE7398ADL\Homework1\Homework1.py
'''
from PIL import Image
from pickle import FALSE
from utils import *
from Model import *
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import sys



def preprocessing(train_test):
    '''
    description: preprocessing the cifar-10 dataset
    param {train_test:decide to return trainloader or testloader}
    return trainloader,testloader
    '''
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #C*H*W
    if train_test=="train":
        set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=FALSE, transform=transform)
        loader = torch.utils.data.DataLoader(set, batch_size=BATCH_SIZE,
                                                shuffle=True, num_workers=0)
    if train_test=="test":
        set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=FALSE, transform=transform)
        loader = torch.utils.data.DataLoader(set, batch_size=BATCH_SIZE,
                                                shuffle=False, num_workers=0)

    return loader

def train(EPOCH,trainloader,testloader,LR):
    
    model=Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    #training
    print_format="{0:<6}{1:<12.4f}{2:<12.4f}{3:<11.4f}{4:<10.4f}" #format accuracy output
    for epoch in range(EPOCH):  # loop over the dataset multiple times

        running_loss = 0.0
        best_loss=10000
        best_acc=0
        correct = 0
        total = 0
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
            _, train_predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (train_predicted == labels).sum().item()
            
            # print statistics
                        
            running_loss += loss.item()
            if i==0:
                print('Loop ','Train Loss ','Train Acc% ','Test Loss ','Test Acc%')
            if i % 200 == 199:    # print every 200 batches
                #test statisctics
                test_correct = 0
                test_total = 0
                test_running_loss=0
                test_running_count=0
                with torch.no_grad():
                    for data in testloader:
                        images, labels = data
                        test_outputs = model(images)
                        test_loss=criterion(test_outputs,labels)
                        test_running_loss+=test_loss.item()
                        test_running_count+=1
                        _, predicted = torch.max(test_outputs.data, 1)
                        test_total += labels.size(0)
                        test_correct += (predicted == labels).sum().item()
                        
                
                acc=100 * test_correct / test_total
                train_acc=100 * correct / total
                print(print_format.format(str(epoch + 1), running_loss / 200,train_acc,test_running_loss/test_running_count,acc))
                
                if running_loss<best_loss and acc>best_acc:
                    best_acc=acc
                    torch.save(model.state_dict(), PATH)
                    
                running_loss = 0.0
                test_running_loss=0.0
                test_running_count=0
    print('train finished')

def validate(img_path):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #C*H*W
    img=Image.open(img_path).convert('RGB')
    img=transform(img).unsqueeze(0) #(1,3,32,32)
    #print(img.shape)
    model=Net()
    model.load_state_dict(torch.load(PATH))
    model.eval()
    output=model(img)
    _, predicted = torch.max(output, 1)

    print('Predicted: ', ' '.join('%5s' % classes[predicted]))

def main():
    if sys.argv[1]=='train':
        trainloader=preprocessing("train")
        testloader=preprocessing("test")
        train(EPOCH=EPOCH,trainloader=trainloader,testloader=testloader,LR=0.01)
    elif sys.argv[1]=='test':
        validate(sys.argv[2])
    else:
        print('wrong parameter')

if __name__ == '__main__':
    main()