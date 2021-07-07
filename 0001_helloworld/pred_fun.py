import torch
import os
import os.path as op
import cv2
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))


def GetLabel(x, y):
    #some function make label 0/1 let network guess
    return x^2+y^2<4


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        #Conv2d(in_channels, out_channels, kernel_size, stride=1,padding=0, dilation=1, groups=1,bias=True, padding_mode=‘zeros’)
        #torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
        self.fun_fc1=nn.Linear(2, 6)
        self.fun_fc2=nn.Linear(6, 2)

    def forward_bak(self, x):
        x = self.fun_fc1(x)
        #x = F.relu(x)
        x = self.fun_fc2(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        # x = torch.flatten(x, 1)
        # x = self.fc1(x)
        # x = F.relu(x)
        # x = self.dropout2(x)
        # x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


model = NeuralNetwork().to(device)
print(model)
#To use the model, we pass it the input data. This executes the model’s forward, along with some background operations. 
# Do not call model.forward() directly!

X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")