# -*- coding:utf-8 -*-
'''
Author: Minghao Chen
Date: 2021-02-08 18:04:35
LastEditTime: 2021-02-17 20:37:00
LastEditors: Please set LastEditors
Description: Classifier models
FilePath: \EECE7398ADL\Homework1\Model.py
'''
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1=nn.Linear(32*32*3, 120)
        self.bn1=nn.BatchNorm1d(120)
        self.ac1=nn.ReLU()
        self.fc2=nn.Linear(120, 84)
        self.bn2=nn.BatchNorm1d(84)
        self.ac2=nn.ReLU()
        self.fc3=nn.Linear(84, 10)

    def forward(self, x):
        x = x.view(-1, 32*32*3)
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.ac1(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.ac2(out)
        out = self.fc3(out)
        return out