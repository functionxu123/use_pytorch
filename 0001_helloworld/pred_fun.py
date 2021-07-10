from numpy.core.defchararray import mod
import torch
import os
import os.path as op
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt



device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

batchsize=100


def GetLabel(x, y):
    #some function make label 0/1 let network guess
    #ret= x**2+y**2<4
    ret=1.5*x-y>0
    #ret=x**2>y
    return ret.astype(int)

def show_param(model, word=""):
    print (word)
    for parameters in model.parameters():
        print(parameters)



class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        #Conv2d(in_channels, out_channels, kernel_size, stride=1,padding=0, dilation=1, groups=1,bias=True, padding_mode=‘zeros’)
        #torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
        self.fun_fc1=nn.Linear(2, 2)
        #self.fun_fc2=nn.Linear(3, 2)
        #self.softm=nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fun_fc1(x)
        #x = F.relu(x)
        #x = self.fun_fc2(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        # x = torch.flatten(x, 1)
        # x = self.fc1(x)
        # x = F.relu(x)
        # x = self.dropout2(x)
        # x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        #output=self.softm(x)
        return output


def train_epoch(model, optimizer,maxstep):
    model.train()

    for i in range(maxstep):
        X = np.random.rand(batchsize, 2).astype(np.float32)
        X=(X-0.5)*6
        
        Y=GetLabel(X[:,0], X[:,1])

        X,Y = torch.from_numpy(X) ,torch.from_numpy(Y)
        X,Y = X.to(device), Y.to(device)
        #print (X,Y)

        optimizer.zero_grad()

        y_pred = model(X)
        #pred_probab = nn.Softmax(dim=1)(logits)
        #y_pred = pred_probab.argmax(1)
        #print(f"Predicted class: {y_pred}")
        loss = F.nll_loss(y_pred, Y)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            #show_param(model)
            print('Train Step: {} Loss: {:.6f}'.format(      i, loss.item()))

def test_epoch(model, sample_cnt=1000):
    model.eval()

    show_param(model,"Showing Testing params:")

    test_loss=0
    correct=0    
    plt.clf()
    plot_size=4
    plt.xlim((-plot_size,plot_size))#设置x轴范围，设置y轴plt.ylim()
    plt.ylim((-plot_size,plot_size))#
    #plt.xticks(np.linspace(-plot_size, plot_size, plot_size*10*2) )
    #plt.yticks(np.linspace(-plot_size, plot_size, plot_size*10*2) )
    plt.grid()
    
    with torch.no_grad():
        for i in range(int(sample_cnt/batchsize)):
            X_np = np.random.rand(batchsize, 2).astype(np.float32)
            X_np=(X_np-0.5)*6
            
            Y_np=GetLabel(X_np[:,0], X_np[:,1])

            X,Y = torch.from_numpy(X_np) ,torch.from_numpy(Y_np)
            X,Y = X.to(device), Y.to(device)
            y_pred = model(X)
            test_loss += F.nll_loss(y_pred, Y, reduction='sum').item()  # sum up batch loss

            pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(Y.view_as(pred)).sum().item()

            pred=pred.cpu().numpy()
            #print (pred)
            draw=X_np[np.where(pred[:,0]==1)]
            #print (y_pred)
            #plt.plot(draw[:,0],draw[:,1],'r')
            #plt.scatter([0], [0], marker = '*',color = 'red', s = 10 )
            plt.scatter(draw[:,0], draw[:,1], marker = '.',color = 'red', s = 5 )
            plt.pause(0.01)
    #plt.show()
    #plt.ioff()
    #plt.close()
            

    total_real=int(sample_cnt/batchsize)*batchsize
    test_loss /= total_real
    print ("Testing: Total: {} AvgLoss: {:.4f} Acc: {:.4f}".format(int(sample_cnt/batchsize)*batchsize, test_loss, correct*1.0/total_real))


def main(model, optimizer, scheduler, maxepoch, eachstep=1000):
    plot_size=4
    #plt.close()
    
    plt.figure(1,figsize=(2*plot_size,2*plot_size))
    plt.ion()
    
    

    for i in range(maxepoch):
        train_epoch(model, optimizer, eachstep)
        test_epoch(model, 3000)

        scheduler.step()
    plt.savefig("result.png")
    plt.close('all')

if __name__ == '__main__':
    model = NeuralNetwork().to(device)
    print(model)
    #To use the model, we pass it the input data. This executes the model’s forward, along with some background operations. 
    # Do not call model.forward() directly!

    optimizer = optim.SGD(model.parameters(), lr=0.0001)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
    main(model, optimizer,scheduler,100, 1000)








