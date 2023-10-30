
from typing import NamedTuple

from torch import Tensor
from torch.nn import (Conv2d,CrossEntropyLoss, Dropout, Flatten, Linear, MaxPool2d, Module,  ReLU, Sequential)
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torchvision.utils import make_grid


import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torch.nn
from torch.autograd import Variable
  

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from torchvision.transforms import (Compose, Normalize, ToTensor)
import onnx
from torch import nn


BATCH_SIZE = 32
transform_mnist = Compose([
        ToTensor(),
        Normalize([0.5], [0.5])
])

img_size = 28 # Image size


# Model
z_size = 100
generator_layer_size = [256, 512, 1024]
discriminator_layer_size = [1024, 512, 256]
num_workers = 8
# Training
epochs = 40  # Train epochs
learning_rate = 1e-4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



mnist_train = (MNIST('./data', train=True, download=True, transform=transform_mnist))
mnist_dataloader = DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)

# real_batch = next(iter(mnist_dataloader))

# plt.figure(figsize=(8,8))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(DEVICE)[:2], padding=2, normalize=True).cpu(),(1,2,0)))

# plt.show()



class_list = ['0','1','2','3','4','5','6','7','8','9']
class_num = len(class_list)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.label_emb = nn.Embedding(10, 10)
        
        self.model = nn.Sequential(
            nn.Linear(794, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, labels):
        x = x.view(x.size(0), 784)
        c = self.label_emb(labels)
        x = torch.cat([x, c], 1)
        out = self.model(x)
        return out.squeeze()
    
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.label_emb = nn.Embedding(10, 10)
        
        self.model = nn.Sequential(
            nn.Linear(110, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )
    
    def forward(self, z, labels):
        z = z.view(z.size(0), 100)
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        out = self.model(x)
        return out.view(x.size(0), 28, 28)
    
def discriminator_train_step(batch_size, discriminator, generator, d_optimizer, criterion, real_images, labels):
    d_optimizer.zero_grad()

    # train with real images
    real_validity = discriminator(real_images, labels)
    real_loss = criterion(real_validity, Variable(torch.ones(batch_size)).cuda())
    
    # train with fake images
    z = Variable(torch.randn(batch_size, 100)).cuda()
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, 10, batch_size))).cuda()
    fake_images = generator(z, fake_labels)
    fake_validity = discriminator(fake_images, fake_labels)
    fake_loss = criterion(fake_validity, Variable(torch.zeros(batch_size)).cuda())
    
    d_loss = real_loss + fake_loss
    d_loss.backward()
    d_optimizer.step()
    return d_loss.item()

def generator_train_step(batch_size, discriminator, generator, g_optimizer, criterion):
    g_optimizer.zero_grad()
    z = Variable(torch.randn(batch_size, 100)).cuda()
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, 10, batch_size))).cuda()
    fake_images = generator(z, fake_labels)
    validity = discriminator(fake_images, fake_labels)
    g_loss = criterion(validity, Variable(torch.ones(batch_size)).cuda())
    g_loss.backward()
    g_optimizer.step()
    return g_loss.item()
def train(generator, discriminator, criterion, d_optimizer, g_optimizer, num_epochs=50, n_critic=5, display_step=50):
    for epoch in range(num_epochs):
        print('Starting epoch {}...'.format(epoch), end=' ')
        for i, (images, labels) in enumerate(mnist_dataloader):
            
            step = epoch * len(mnist_dataloader) + i + 1
            real_images = Variable(images).cuda()
            labels = Variable(labels).cuda()
            generator.train()
            
            d_loss = 0
            for _ in range(n_critic):
                d_loss = discriminator_train_step(len(real_images), discriminator,
                                                  generator, d_optimizer, criterion,
                                                  real_images, labels)

            g_loss = generator_train_step(BATCH_SIZE, discriminator, generator, g_optimizer, criterion)
            print(f"{i}/{BATCH_SIZE}", end=' ')
            print('D: {:.4f}, G: {:.4f}'.format(d_loss, g_loss), end=' ')
            
            # if step % display_step == 0:
            #     generator.eval()
            #     z = Variable(torch.randn(9, 100)).cuda()
            #     labels = Variable(torch.LongTensor(np.arange(9))).cuda()
            #     sample_images = generator(z, labels).unsqueeze(1)
            #     grid = make_grid(sample_images, nrow=3, normalize=True)
            #     plt.figure(figsize=(10, 10))
            #     plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
            #     plt.axis('off')
            #     plt.show()
                
        print('Done!')

def generate_digit(generator, digit, num_images=1):
    z = Variable(torch.randn(num_images, 100)).cuda()
    label = torch.LongTensor([digit] * num_images).cuda()
    imgs = generator(z, label).data.cpu()
    imgs = 0.5 * imgs + 0.5
    pil_imgs = [transforms.ToPILImage()(img) for img in imgs]
    
    return pil_imgs
if __name__ == '__main__':
    generator = Generator().cuda()
    discriminator = Discriminator().cuda()
    criterion = nn.BCELoss()
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
        
    #train(generator=generator, discriminator=discriminator, criterion=criterion, d_optimizer=d_optimizer, g_optimizer=g_optimizer)
    #torch.save(generator.state_dict(), 'cGAN_mnist_generator.pt')

   
    generator.load_state_dict(torch.load("cGAN_mnist_generator.pt"))
    generator.eval()
    dummy_input_z = Variable(torch.randn(1, 100)).cuda()
    dummy_input_label = Variable(torch.LongTensor([0])).cuda()
    z = Variable(torch.randn(1, 100)).cuda()
    # labels = torch.LongTensor([i for i in range(10) for _ in range(10)]).cuda()
    # images = generator(z, labels).unsqueeze(1)
    # grid = make_grid(images, nrow=10, normalize=True)
    # fig, ax = plt.subplots(figsize=(10, 10))
    # ax.imshow(grid.permute(1, 2, 0).data.cpu(), cmap='binary')
    # ax.axis('off')
    # plt.show()

        # Export the trained model to ONNX
    input_names = ["inputTensor", "inputLabel"]  # or some other name for your input
    output_names = ["output"]  # or some other name for your output
    torch.onnx.export(generator, (dummy_input_z, dummy_input_label), "mnist_model_cGan.onnx", verbose=True, input_names=input_names, output_names=output_names)


    onnx_model = onnx.load("mnist_model_cGan.onnx")
    onnx.checker.check_model(onnx_model)

    # Print a human-readable representation of the graph
    onnx.helper.printable_graph(onnx_model.graph)
    # imgs = generate_digit(generator, 4, 5)
    # for img in imgs:
    #     plt.imshow(img, cmap='gray')
    #     plt.show()
