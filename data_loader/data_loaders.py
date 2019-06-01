from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from base import BaseDataLoader

import six
import lmdb
import msgpack
from PIL import Image
import os.path as osp


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor()
        ])
        self.data_dir = data_dir

        class MNIST(datasets.MNIST):
            def __getitem__(self, index):
                sample = super().__getitem__(index)[0]
                return sample

        self.dataset = MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super(MnistDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class ImageFolderLMDB(Dataset):
    def __init__(self, db_path, transform=None, target_transform=None):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.get(b'__len__')
            self.keys = msgpack.loads(txn.get(b'__keys__'))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = None, None
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        unpacked = msgpack.loads(byteflow)

        # load image
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'


class _no_class_imagefolder(datasets.ImageFolder):
    def __getitem__(self, index):
        sample = super().__getitem__(index)[0]
        return sample


class AnimeFaceDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, resize=None, shuffle=True, validation_split=0.0, num_workers=1):
        trsfm = [transforms.RandomHorizontalFlip(), transforms.ToTensor()]
        if resize:
            trsfm = [transforms.Resize(resize)] + trsfm
        trsfm = transforms.Compose(trsfm)
        self.data_dir = data_dir
        self.dataset = _no_class_imagefolder(self.data_dir, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


if __name__ == '__main__':
    loader = ImageFolderLMDB('/media/ycy/86A4D88BA4D87F5D/DataSet/animation_lmdb/animation.lmdb', 256)
    for batch_idx, data in enumerate(loader):
        print(data.shape)
