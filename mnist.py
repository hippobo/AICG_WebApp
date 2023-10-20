
from typing import NamedTuple

from torch import Tensor
from torch.nn import (Conv2d,CrossEntropyLoss, Dropout, Flatten, Linear, MaxPool2d, Module,  ReLU, Sequential)
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torch.nn
  
from torchvision.transforms import (Compose, Lambda, RandomRotation, ToTensor)



batch_size = 512
transform_mnist = Compose([ToTensor()])
mnist_train = MNIST('./data', train=True, download=True, transform=transform_mnist)
mnist_test = MNIST('./data', train=False, download=True, transform=transform_mnist)
train_dataloader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=8)
test_dataloader = DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=8)

class Loaders(NamedTuple):
  train : DataLoader
  test : DataLoader

loaders = Loaders(train=train_dataloader, test=test_dataloader) 



class CNN(Module):
    def __init__(self) -> None:
        super(CNN, self).__init__()
        self.conv1 = Sequential(         
            Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            ReLU(),                      
            MaxPool2d(kernel_size=2),    
        )
        self.conv2 = Sequential(         
            Conv2d(16, 32, 5, 1, 2),     
            ReLU(),                      
            MaxPool2d(2),                
        )
        
        self.out = Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output    

def step(model , optimizer : torch.optim, loss_func,  loaders : Loaders, device : str, train : bool = True):
    loader = loaders.train if train else loaders.test
    model = model.train(mode = train)

    avg_loss, avg_acc = 0.0, 0.0

    for batch, labels in loader:
        batch, labels = batch.to(device), labels.to(device)
        y_hat = model(batch)
        loss =  loss_func(y_hat,labels)
        
        acc = (y_hat.argmax(dim = 1) == labels).sum()

        
        

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        avg_acc += acc.item()
        avg_loss += loss.item()
        

    avg_acc /= len(loader.dataset)
    avg_loss /= len(loader)


    return avg_loss, avg_acc




model = CNN().to("cuda")
epochs = 10
optim = AdamW(params=model.parameters(), lr=0.01)
best_acc = 0
train_acc_hist, test_acc_hist, train_loss_hist, test_loss_hist = [], [], [], []

for epoch in range(epochs):
    train_loss, train_acc = step(model=model, optimizer=optim, loss_func=F.cross_entropy, loaders=loaders, device="cuda")
    test_loss, test_acc = step(model=model, optimizer=optim, loss_func=F.cross_entropy, loaders=loaders, device="cuda", train=False)

    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), "mnist_model.pt")
    print("Epoch:", epoch + 1)
    print(f"train loss:{train_loss:2f}")
    print(f"test loss:{test_loss:2f}")
    print(f"{train_acc * 100:.2f}%")
    print(f"{test_acc * 100:.2f}%")
    train_loss_hist.append(train_loss)
    test_loss_hist.append(test_loss)
    train_acc_hist.append(train_acc)
    test_acc_hist.append(test_acc)


fig, ax = plt.subplots(1,2, figsize=(10,5))
print(f"Best Test Accuracy{best_acc * 100}%")

ax[0].plot(np.arange(epochs), train_loss_hist, label="train loss")
ax[0].plot(np.arange(epochs), test_loss_hist, label="test loss")
ax[0].set_xlabel("epochs")
ax[0].set_ylabel("loss")
ax[0].legend()

ax[1].plot(np.arange(epochs), train_acc_hist, label="train acc")
ax[1].plot(np.arange(epochs), test_acc_hist, label="test acc")
ax[1].set_xlabel("epochs")
ax[1].set_ylabel("accuracy")
ax[1].legend()

plt.legend()
plt.show()
   






