import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
import time
from configuration.config import Configuration
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import torch.nn.init as init

class ShadowModel(nn.Module):
    def __init__(self,in_channels, out_channels, fc_size):
        super(ShadowModel, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels[0], kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(out_channels[0], out_channels=out_channels[1], kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Linear(fc_size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, sample):
        out = self.conv(sample)
        out = self.fc(out.view(sample.size(0), -1))
        out = torch.softmax(out, dim=1)
        return out

class Encoder(nn.Module):
    def __init__(self, mu_dim, num_imgs,leaky_relu_param, dropout_params):
        super(Encoder, self).__init__()
        self.num_imgs = num_imgs
        self.fc1 = nn.Linear(1000, 128)
        self.fc2 = nn.Linear(128, mu_dim)
        self.leaky_relu=leaky_relu_param
        self.dropout=dropout_params

    def forward (self, delta):
        out = torch.dropout(F.leaky_relu(self.fc1(delta), self.leaky_relu[0]), self.dropout[0], self.training)
        mu = torch.dropout(F.leaky_relu(self.fc2(out), self.leaky_relu[1]), self.dropout[1], self.training)
        mu = mu.repeat(self.num_imgs, 1)
        return mu




class Generator(nn.Module):
    def __init__(self, mu_size,noise_size, leaky_relu, ngf):
        super(Generator, self).__init__()
        #self.image_size = image_size
        self.input_size=mu_size+noise_size
        self.leaky_relu=leaky_relu
        self.ngf=ngf

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(self.input_size, self.ngf, 2, 1, 0, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.LeakyReLU(self.leaky_relu, inplace=False),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf, self.ngf//2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf//2),
            nn.LeakyReLU(self.leaky_relu, inplace=False),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.ngf//2, self.ngf//4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf//4),
            nn.LeakyReLU(self.leaky_relu, inplace=False),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(self.ngf//4, self.ngf//8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf//8),
            nn.LeakyReLU(self.leaky_relu, inplace=False),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(self.ngf//8, 3, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, mu, z):
        x=torch.cat([mu,z], dim=1)
        gen_images=self.main(x)
        return gen_images


class Discriminator(nn.Module):
    def __init__(self, leaky_relu, ndf):
        super(Discriminator, self).__init__()
        self.leaky_relu=leaky_relu
        self.ndf=ndf
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(3, self.ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf),
            nn.LeakyReLU(self.leaky_relu, inplace=False),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(self.leaky_relu, inplace=False),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(self.leaky_relu, inplace=False),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(self.leaky_relu, inplace=False),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8, 1, 2, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input,mu):
        return self.main(input)


class MinibatchDiscrimination(nn.Module):
    def __init__(self, in_features, out_features, kernel_dims, mean=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims
        self.mean = mean
        self.T = nn.Parameter(torch.Tensor(in_features, out_features, kernel_dims)).to("cuda:0")
        init.normal(self.T, 0, 1)

    def forward(self, x):
        # x is NxA
        # T is AxBxC
        matrices = x.mm(self.T.view(self.in_features, -1))
        matrices = matrices.view(-1, self.out_features, self.kernel_dims)

        M = matrices.unsqueeze(0)  # 1xNxBxC
        M_T = M.permute(1, 0, 2, 3)  # Nx1xBxC
        norm = torch.abs(M - M_T).sum(3)  # NxNxB
        expnorm = torch.exp(-norm)
        o_b = (expnorm.sum(0) - 1)   # NxB, subtract self distance
        if self.mean:
            o_b /= x.size(0) - 1

        x = torch.cat([x, o_b], 1)
        return x

class NewDiscriminator(nn.Module):
    def __init__(self, leaky_relu, dropput):
        super(NewDiscriminator, self).__init__()
        self.dropout=dropput
        self.leaky_relu=leaky_relu
        self.conv1=spectral_norm(nn.Conv2d(3, 64, kernel_size=4,stride= 2))
        self.conv2=spectral_norm(nn.Conv2d(64, 128, kernel_size=4,stride= 2))
        self.conv3 = spectral_norm(nn.Conv2d(128, 256, kernel_size=4, stride=2))
        self.fc1=spectral_norm(nn.Linear(1024+64+64, 512))
        self.fc2 = spectral_norm(nn.Linear(512, 64))
        self.fc3 = spectral_norm(nn.Linear(64, 16))
        self.fc4 = spectral_norm(nn.Linear(16, 1))

    def forward(self, image, mu):
        x=image
        x=nn.LeakyReLU(0.2)(self.conv1(x))
        x=nn.LeakyReLU(0.2)(self.conv2(x))
        x=nn.LeakyReLU(0.2)(self.conv3(x))
        x=x.reshape(x.shape[0],-1)
        x=torch.cat([x,mu.reshape(mu.shape[0],-1)], dim=1)
        x=MinibatchDiscrimination(1088, 64, 16)(x)
        x=nn.Dropout(self.dropout)(nn.LeakyReLU(0.2)(self.fc1(x)))
        x = nn.Dropout(self.dropout)(nn.LeakyReLU(0.2)(self.fc2(x)))
        x = nn.Dropout(self.dropout)(nn.LeakyReLU(0.2)(self.fc3(x)))
        x=self.fc4(x)
        return nn.Sigmoid()(x)



class NewGenerator(nn.Module):
    def __init__(self, mu_size,noise_size):
        super(NewGenerator, self).__init__()
        #self.image_size = image_size
        self.input_size=mu_size+noise_size

        self.fc = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128,512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(4096)
        )
        self.conv=nn.Sequential(
            nn.ConvTranspose2d(256, 128,kernel_size=4, stride=2,padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2,padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2,padding=1),
            nn.Tanh()
        )

    def forward(self, mu, z):

        x=torch.cat([mu,z], dim=1)

        size=x.shape[0]
        x=x.reshape(size,-1)
        x=self.fc(x)
        x=x.reshape(size,256,4,4)
        x=self.conv(x)
        return x