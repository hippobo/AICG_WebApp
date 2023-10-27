
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
from torchvision.transforms import (Compose, Lambda, RandomRotation, ToTensor)
import onnx
from torch import nn


BATCH_SIZE = 128
transform_mnist = Compose([ToTensor()])
NUM_WORKERS = 8
image_size = 28
nc = 1
nz = 100
ngf = 32
ndf = 32
ngpu = 1
lr = 0.002
beta1 = 0.5
num_epochs = 30
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



mnist_train = (MNIST('./data', train=True, download=True, transform=transform_mnist))
mnist_dataloader = DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

# real_batch = next(iter(mnist_dataloader))

# plt.figure(figsize=(8,8))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(DEVICE)[:2], padding=2, normalize=True).cpu(),(1,2,0)))

# plt.show()

# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    

class ConditionalGenerator(nn.Module):
     def __init__(self, ngpu, n_classes):
        super(ConditionalGenerator, self).__init__()
        self.ngpu = ngpu
        self.label_emb = nn.Embedding(n_classes, nz) 
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 4, 3, 1, 0, bias=False),  # new layer
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 3 x 3
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 0, bias=False),  # modified
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 7 x 7
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),  # modified
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 14 x 14
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),  # modified
            nn.Tanh()
            # state size. (nc) x 28 x 28
        )

     def forward(self, noise, labels):
        # Concatenate label embedding with noise
        gen_input = torch.mul(self.label_emb(labels), noise)
        # gen_input shape should match the expected input shape for your architecture
        return self.main(gen_input)

    
netG = ConditionalGenerator(ngpu, 10).to(DEVICE)



class ConditionalDiscriminator(nn.Module):
    def __init__(self, ngpu, n_classes):
        super(ConditionalDiscriminator, self).__init__()
        self.ngpu = ngpu
        self.label_emb = nn.Embedding(n_classes, 50)  # for MNIST
        self.main = nn.Sequential(
            # input is (nc) x 28 x 28
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 14 x 14
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 7 x 7
            nn.Conv2d(ndf * 2, 1, 7, 1, 0, bias=False),  # modified
            nn.Sigmoid()
            # state size. 1 x 1 x 1
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image
        d_in = torch.cat((img.view(img.size(0), -1), self.label_emb(labels)), -1)
        # You must reshape d_in to match the expected input shape for your architecture
        return self.main(d_in.view(img.size(0), nc, image_size, image_size))

netD = ConditionalDiscriminator(ngpu, 10).to(DEVICE)
    # Initialize the ``BCELoss`` function
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
num_classes = 10
fixed_noise = torch.randn(64, nz, 1, 1, device=DEVICE)
fixed_labels = torch.randint(0, num_classes, (64,), dtype=torch.long, device=DEVICE)  # Random fixed labels


# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# Setup Adam optimizers for both G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Training Loop

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0
print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(mnist_dataloader, 0):
        # Get the inputs and the labels from the data tuple
        inputs, labels = data
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        b_size = inputs.size(0)

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        netD.zero_grad()
        
        # Forward pass real batch through D
        output = netD(inputs, labels).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, torch.full((b_size,), real_label, device=DEVICE))
        errD_real.backward()
        D_x = output.mean().item()

        # Generate fake image batch with G
        noise = torch.randn(b_size, nz, 1, 1, device=DEVICE)
        random_labels = torch.randint(0, num_classes, (b_size,), dtype=torch.long, device=DEVICE)
        fake = netG(noise, random_labels)
        label = torch.full((b_size,), fake_label, device=DEVICE)

        # Classify all fake batch with D
        output = netD(fake.detach(), random_labels).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        errD_fake.backward()
        D_G_z1 = output.mean().item()

        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()

        label.fill_(real_label)  # fake labels are real for generator cost
        # Since D just updated, perform another forward pass of all-fake batch through D
        output = netD(fake, random_labels).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()

        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(mnist_dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(mnist_dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise, fixed_labels).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1


plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="G")
plt.plot(D_losses,label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Grab a batch of real images from the dataloader
real_batch = next(iter(mnist_dataloader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(DEVICE)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()