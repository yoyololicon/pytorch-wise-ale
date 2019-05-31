import torch
import torch.nn.functional as F
import math

pi2 = 2 * math.pi
logpi2 = math.log(pi2)
eps = 1e-45


def _upack_and_image_loss(output, target):
    z, logvar, mu, recon = output
    # x.shape = (batch, channels, H, W)
    # recon_x.shape = (batch, L, channels, H, W)
    # z.shape = (batch, L, K)
    # mu and logvar.shape = (batch, K)
    L = z.shape[1]
    BCE = F.binary_cross_entropy_with_logits(recon, target.unsqueeze(1).expand_as(recon), reduction='sum') / L
    return z, logvar, mu, BCE


# Reconstruction + KL divergence losses summed over all elements and batch
def AEVB(output, target):
    z, logvar, mu, BCE = _upack_and_image_loss(output, target)
    batch, L, K = z.shape
    KLD = 0.5 * (mu.pow(2).sum() + logvar.exp().sum() - logvar.sum() - batch * K)
    return (BCE + KLD) / batch


def WiSE_MM(output, target):
    z, *_, BCE = _upack_and_image_loss(output, target)
    batch, L, K = z.shape

    z = z.view(batch * L, K)
    mu = z.mean(0)
    m = z - mu
    cov = m.t() @ m / (batch * L - 1)

    KLD = 0.5 * (mu @ mu + cov.trace() - cov.slogdet()[1] - K)  # det() is zero, need fixed
    return BCE / batch + KLD


def WiSE_UB(output, target):
    z, logvar, mu, BCE = _upack_and_image_loss(output, target)
    batch, L, K = z.shape

    var = logvar.exp()

    C = (var.sum() + mu.pow(2).sum() + logpi2 * batch * K) * 0.5 / batch
    var_add = var[:, None] + var[None, ...]  # (batch, batch, k)
    mu_minus_square = torch.pow(mu[:, None] - mu[None, ...], 2)

    logAB = -0.5 * (var_add.log().add(logpi2).sum(2) + (mu_minus_square / var_add).sum(2))
    logAB_max = logAB.max(0)[0]
    logAB = logAB - logAB_max
    AB = logAB.exp().mean(0).log().mean() + logAB_max.mean()

    KLD = AB + C
    return BCE / batch + KLD


def WiSE_UB2(output, target):
    z, logvar, mu, BCE = _upack_and_image_loss(output, target)
    batch, L, K = z.shape

    var = logvar.exp()

    C = (var.sum() + mu.pow(2).sum() + logpi2 * batch * K) * 0.5 / batch
    var_add = var[:, None] + var[None, ...]  # (batch, batch, k)
    mu_minus_square = torch.pow(mu[:, None] - mu[None, ...], 2)

    logAB = -0.5 * (var_add.log().add(logpi2).sum(2) + (mu_minus_square / var_add).sum(2))
    logAB_max = logAB.max()
    logAB = logAB - logAB_max
    AB = logAB.exp().mean().log() + logAB_max

    KLD = AB + C
    return BCE / batch + KLD


if __name__ == '__main__':
    mu = torch.randn(1, 64) + torch.randn(5, 1) * 0.001
    logvar = torch.rand(5, 64)
    output = (
        torch.nn.Parameter(torch.randn(5, 1, 64)) * 0.0001 + mu[:, None, :], logvar, mu, torch.randn(5, 1, 3, 64, 64))
    target = torch.randn(5, 3, 64, 64)

    print(AEVB(output, target), WiSE_MM(output, target), WiSE_UB(output, target), WiSE_UB2(output, target))
