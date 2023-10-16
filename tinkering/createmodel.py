import os
import torch
from torch import nn
import torch.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
print(f"Using {device} device")

class myconvnet(nn.Module):
    def __init__(self, input_channels, out_vector,out_shape):
        super(myconvnet, self).__init__()
        self.out_shape = out_shape
        self.conv1 =  torch.nn.Conv2d(input_channels,8,3)
        self.conv2 = torch.nn.Conv2d(8, 16, 3)
        self.mp1 = torch.nn.MaxPool2d(2)
        self.conv3 = torch.nn.Conv2d(16, 32, 3)
        self.conv4 = torch.nn.Conv2d(32, 64, 3)
        self.mp3 = torch.nn.MaxPool2d(3)
        self.conv5 = torch.nn.Conv2d(64, 64, 3)
        self.conv6 = torch.nn.Conv2d(64, 32, 3)
        self.conv7 = torch.nn.Conv2d(32, 32, 3)
        self.conv8 = torch.nn.Conv2d(32, 16, 1)
        self.fc1 = torch.nn.Linear(2304, 800)
        self.fc2 = torch.nn.Linear(800, out_vector)
        self.rel = nn.ReLU(inplace=False)
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        # Max pooling over a (2, 2) window
        x = self.mp1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x=self.mp3(x)
        x=self.conv5(x)
        x=self.conv6(x)
        # x = self.mp1(x)
        x=self.mp3(x)
        x=self.conv7(x)
        x = self.conv8(x)
        x= torch.flatten(x)
        x= self.rel(self.fc1(x))
        x=self.rel(self.fc2(x))
        x=torch.reshape(x,self.out_shape)
        return x

    def get_model_params(self,model):
        print(f"Model structure: {model}\n\n")

        for name, param in model.named_parameters():
            print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")