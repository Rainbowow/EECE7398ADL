'''
Author: your name
Date: 2021-02-28 13:35:37
LastEditTime: 2021-02-28 15:42:18
LastEditors: Please set LastEditors
Description: CNN model using MobileNet V1
FilePath: \Homework2\Model.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, 5),stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.conv_dw1=self.block(32,64,1)
        self.conv_dw2=self.block(64,128,2)
        self.conv_dw3=self.block(128,128,1)
        self.conv_dw4=self.block(128,256,2)
        self.conv_dw5=self.block(256,256,1)
        self.conv_dw6=self.block(256,512,2)
        self.layers = self._dw_layers(512,512,5)
        self.conv_dw7=self.block(512,1024,2)
        self.conv_dw8=self.block(1024,1024,1)
        self.avgpool=nn.AvgPool2d(kernel_size=2,stride=1)
        self.fc1=nn.Linear(1024,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)

    def forward(self, x):
        out=self.conv1(x)
        out=self.conv_dw1(out)
        out=self.conv_dw2(out)
        out=self.conv_dw3(out)
        out=self.conv_dw4(out)
        out=self.conv_dw5(out)
        out=self.conv_dw6(out)
        out=self.layers(out)
        out=self.conv_dw7(out)
        out=self.conv_dw8(out)
        out=self.avgpool(out)
        out=out.view(-1,out.size(1))
        out=self.fc1(out)
        out=self.fc2(out)
        out=self.fc3(out)
        return out

    def _dw_layers(self,in_channels,out_channels,blocks):
        layers=[]
        for i in range(blocks):
            layers.append(self.block(in_channels,out_channels,1))
        return nn.Sequential(*layers)
    def block(self,in_channels, out_channels, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

# net = MobileNetV1()
# x = torch.rand(1,3,32,32)
# for name,layer in net.named_children():
#     if (name == "fc1") or (name == "fc2") or (name == "fc3"):
#         x = x.view(-1,x.size(1))
#         x = layer(x)
#         print(name, 'output shape:', x.shape)
#     else:
#         x = layer(x)
#         print(name, 'output shape:', x.shape)

'''
conv1 output shape: torch.Size([1, 32, 28, 28])
conv_dw1 output shape: torch.Size([1, 64, 28, 28])
conv_dw2 output shape: torch.Size([1, 128, 14, 14])
conv_dw3 output shape: torch.Size([1, 128, 14, 14])
conv_dw4 output shape: torch.Size([1, 256, 7, 7])
conv_dw5 output shape: torch.Size([1, 256, 7, 7])
conv_dw6 output shape: torch.Size([1, 512, 4, 4])
layers output shape: torch.Size([1, 512, 4, 4])
conv_dw7 output shape: torch.Size([1, 1024, 2, 2])
conv_dw8 output shape: torch.Size([1, 1024, 2, 2])
avgpool output shape: torch.Size([1, 1024, 1, 1])
fc1 output shape: torch.Size([1, 120])
fc2 output shape: torch.Size([1, 84])
fc3 output shape: torch.Size([1, 10])
'''