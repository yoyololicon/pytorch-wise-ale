import argparse
import torch
from torchvision.utils import make_grid
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kde


def main(config, resume, size=16, z_range=3):
    # build model architecture
    model = config.initialize('arch', module_arch)
    data_loader = module_data.MnistDataLoader(config.config['data_loader']['args']['data_dir'], size, False)

    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            _, logvar, mu = model(data, decode=False)
            std, mu = logvar.squeeze().exp().sqrt().cpu().numpy(), mu.squeeze().cpu().numpy()
            break

    for i in range(size):
        samples = np.random.randn(200, 2) * std[i] + mu[i]
        k = kde.gaussian_kde(samples.T)
        x, y = samples.T
        xi, yi = np.mgrid[x.min():x.max():20 * 1j, y.min():y.max():20 * 1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))

        plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap=plt.cm.BuGn_r)
        plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-t', default=5, type=float)
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')

    args = parser.parse_args()
    config = ConfigParser(parser)
    torch.manual_seed(args.seed)
    main(config, args.resume)
