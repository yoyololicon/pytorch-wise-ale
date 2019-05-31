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
        # self.thresh = math.pow(3.4e+38, -2 / hidden_dim) * 0.5 * math.pi  # make sure not overflow in WiSE UB
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


class _mapping(nn.Module):
    def __init__(self, in_channels, out_channels, layers=3):
        super().__init__()
        nets = []
        for _ in range(layers):
            nets += [nn.Linear(in_channels, out_channels), nn.LeakyReLU(0.2, inplace=True)]
            in_channels = out_channels
        self.f = nn.Sequential(*nets)

    def forward(self, x):
        return self.f(x)


class _AdaIN(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()
        self.in_channel = in_channel
        self.w = nn.Linear(style_dim, in_channel * 2)
        self.w.weight.data.fill_(0)
        self.w.bias.data[:in_channel] = 1
        self.w.bias.data[in_channel:] = 0

    def forward(self, input, style):
        y_s, y_b = self.w(style)[..., None, None].split(self.in_channel, 1)
        return y_s * F.instance_norm(input, eps=1e-8) + y_b


class _noise(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1))

    def forward(self, x, noise):
        return x + self.weight * noise


class _style_block(nn.Module):
    def __init__(self, in_channel, out_channel, style_dim, kernel_size, padding, bias=False):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, bias=bias)
        # self.noise1 = _noise(out_channel)
        self.adain1 = _AdaIN(out_channel, style_dim)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size, padding=padding, bias=bias)
        # self.noise2 = _noise(out_channel)
        self.adain2 = _AdaIN(out_channel, style_dim)

    def forward(self, input, style, noise=None):
        # probably not gonna add noise, as we sample from a prior distribution is already very blury
        # if noise is None:
        #    noise = torch.randn_like(input[:, :1])
        # else:
        #    noise = noise[:, None]
        out = self.conv1(input)
        # out = self.noise1(out, noise)
        out = F.leaky_relu(self.adain1(out, style), 0.2, inplace=True)

        out = self.conv2(out)
        # out = self.noise2(out, noise)
        out = F.leaky_relu(self.adain2(out, style), 0.2, inplace=True)
        return out


class _stylegan_decoder(nn.Module):
    def __init__(self, input_size=64, hidden_dim=64):
        super().__init__()
        self.reshape_size = (1024, input_size // 8, input_size // 8)
        self.constant_input = nn.Parameter(torch.ones(1, *self.reshape_size))
        self.map_func = _mapping(hidden_dim, hidden_dim)

        self.dec_convs = nn.ModuleList([_style_block(1024, 512, hidden_dim, 3, 1),
                                        _style_block(512, 256, hidden_dim, 3, 1),
                                        _style_block(256, 128, hidden_dim, 3, 1),
                                        _style_block(128, 64, hidden_dim, 3, 1)])

        self.final_conv = nn.Conv2d(64, 3, 5, padding=2)

    def forward(self, z):
        batch_size, *_ = z.shape
        z = self.map_func(z)

        x = self.constant_input.expand(batch_size, -1, -1, -1)
        for i, conv in enumerate(self.dec_convs):
            if i:
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = conv(x, z)

        return self.final_conv(x)


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


if __name__ == '__main__':
    net = VAE(64)
    import numpy as np

    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    print(sum([np.prod(p.size()) for p in model_parameters]))

    x = torch.randn(5, 3, 64, 64)
    z, logvar, mu, x2 = net(x, 3, True)
    print(z.shape, logvar.shape, mu.shape, x2.shape)
