import torch
import torch.nn.functional as F


def my_metric(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def my_metric2(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def kl_div(output, target):
    with torch.no_grad():
        z, logvar, mu, recon = output
        #batch, L, K = z.shape
        KLD = 0.5 * (mu.pow(2).mean() + logvar.exp().mean() - logvar.mean() - 1)
    return KLD


def reconstruct(output, target):
    *_, recon = output
    batch, L, *_ = recon.shape
    with torch.no_grad():
        return F.binary_cross_entropy_with_logits(recon, target.unsqueeze(1).expand_as(recon))


def ELBO(output, target):
    z, var, mu, recon = output
    batch, L, *_ = recon.shape
    target = target.unsqueeze(1).expand_as(recon)
    log_recon = - torch.log(1 + torch.exp(-recon))
    likelihood = target.view(-1) @ log_recon.view(-1) / L / batch
    elbo = likelihood - kl_div(output) * z.shape[-1]
    return elbo

