import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel

import math


class _Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.shape[0], -1)


class _encoder(nn.Module):
    def __init__(self, input_size=64, hidden_dim=64):
        super().__init__()
        self.dim = hidden_dim
        self.enc_convs = nn.Sequential(nn.Conv2d(3, 128, 5, padding=2, stride=2, bias=False),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(128, 256, 5, padding=2, stride=2, bias=False),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(256, 512, 5, padding=2, stride=2, bias=False),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(512, 1024, 5, padding=2, stride=2, bias=False),
                                       nn.ReLU(inplace=True),
                                       _Flatten(),
                                       nn.Linear(1024 * (input_size // 16) ** 2, hidden_dim * 2))

    def forward(self, x):
        return torch.split(self.enc_convs(x), self.dim, -1)


class _smooth_decoder(nn.Module):
    def __init__(self, input_size=64, hidden_dim=64):
        super().__init__()
        self.reshape_size = (1024, input_size // 8, input_size // 8)
        self.linear = nn.Linear(hidden_dim, 1024 * (input_size // 8) ** 2, bias=False)
        self.dec_convs = \
            nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                          nn.Conv2d(1024, 512, 5, padding=2, bias=False),
                          nn.BatchNorm2d(512),
                          nn.ReLU(inplace=True),
                          nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                          nn.Conv2d(512, 256, 5, padding=2, bias=False),
                          nn.BatchNorm2d(256),
                          nn.ReLU(inplace=True),
                          nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                          nn.Conv2d(256, 128, 5, padding=2, bias=False),
                          nn.BatchNorm2d(128),
                          nn.ReLU(inplace=True),
                          nn.Conv2d(128, 3, 5, padding=2))

    def forward(self, z):
        z = F.relu(self.linear(z), inplace=True).view(-1, *self.reshape_size)
        return self.dec_convs(z)


class VAE(BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.enc = _encoder(*args, **kwargs)
        self.dec = _smooth_decoder(*args, **kwargs)

    def sampling(self, logvar, mu):
        return torch.randn_like(mu) * logvar.mul(0.5).exp() + mu

    def forward(self, x=None, L=1, decode=True, z=None):
        if z is None:
            batch, *dims = x.shape
            logvar, mu = self.enc(x)
            z = self.sampling(logvar.unsqueeze(1), mu.unsqueeze(1).expand(-1, L, -1))
            if decode:
                return z, logvar, mu, self.dec(z.view(batch * L, -1)).view(batch, L, *dims)
            else:
                return z, logvar, mu
        else:
            return self.dec(z)


class _small_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_convs = nn.Sequential(nn.ConstantPad1d((2, 1), 0.),
                                       nn.Conv2d(1, 16, 4, stride=2, bias=False),
                                       nn.ReLU(inplace=True),
                                       nn.ConstantPad1d((2, 1), 0.),
                                       nn.Conv2d(16, 32, 4, stride=2, bias=False),
                                       nn.ReLU(inplace=True),
                                       nn.ConstantPad1d(1, 0.),
                                       nn.Conv2d(32, 64, 4, stride=2, bias=False),
                                       nn.ReLU(inplace=True),
                                       _Flatten())
        self.mu_linear = nn.Sequential(nn.Linear(64 * 3, 32),
                                       nn.Linear(32, 2))
        self.logvar_linear = nn.Linear(64 * 3, 2)

    def forward(self, x):
        x = self.enc_convs(x)
        return self.logvar_linear(x), self.mu_linear(x)


class _small_decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dec_linears = nn.Sequential(nn.Linear(2, 16),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(16, 128),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(128, 784))

    def forward(self, z):
        return self.dec_linears(z).view(-1, 1, 28, 28)


class MNIST_VAE(VAE):
    def __init__(self):
        BaseModel.__init__(self)
        self.enc = _small_encoder()
        self.dec = _small_decoder()


if __name__ == '__main__':
    net = MNIST_VAE()
    import numpy as np

    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    print(sum([np.prod(p.size()) for p in model_parameters]))

    x = torch.randn(5, 1, 28, 28)
    z, logvar, mu, x2 = net(x, 3, True)
    print(z.shape, logvar.shape, mu.shape, x2.shape)
