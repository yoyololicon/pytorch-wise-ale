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


def main(config, resume, size=10, z_range=3):
    # build model architecture
    model = config.initialize('arch', module_arch)

    checkpoint = torch.load(resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    latent_code = torch.stack(
        torch.meshgrid(torch.linspace(-z_range, z_range, size), torch.linspace(-z_range, z_range, size)), 2).view(-1,
                                                                                                                  2).cuda()

    with torch.no_grad():
        output = model(z=latent_code).view(size, size, 28, 28).sigmoid().cpu().permute(0, 2, 1, 3).contiguous().view(
            28 * size, -1).numpy()
        plt.imshow(output, aspect='auto')
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
