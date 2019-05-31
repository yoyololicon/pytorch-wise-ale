import argparse
import torch
from torchvision.utils import make_grid
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser

import os
import numpy as np
from skvideo.io import vwrite


def gif(filename, image_list):
    fname, _ = os.path.splitext(filename)
    filename = fname + '.mp4'

    np_images = []
    for m in image_list:
        np_images.append(m.permute(1, 2, 0).numpy() * 255)

    vwrite(filename, np_images, verbosity=0)
    return


def main(config, resume, seconds, filename):


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

    batch_size = 16
    hidden_dim = config['arch']['args']['hidden_dim']
    start_code = torch.randn(batch_size, hidden_dim).to(device)
    end_code = torch.randn(batch_size, hidden_dim).to(device)
    fps = 24
    total_steps = int(fps * seconds)

    output_list = []
    with torch.no_grad():
        for i in torch.linspace(0, 1, total_steps):
            code = i * end_code + (1 - i) * start_code
            output = model(z=code).sigmoid().cpu()
            output_list.append(make_grid(output, nrow=4))

    gif(filename, output_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('filename', type=str)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('-t', default=5, type=float)
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    parser.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')

    args = parser.parse_args()
    config = ConfigParser(parser)
    torch.manual_seed(args.seed)
    main(config, args.resume, args.t, args.filename)
