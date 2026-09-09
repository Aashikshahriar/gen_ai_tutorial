"""
Manual MNIST loader — no torchvision.datasets, no sklearn.fetch_openml.

Downloads the raw IDX-ubyte(.gz) files directly and parses the binary format by hand:

    IDX image file header (big-endian):
        magic number (4 bytes) = 2051
        num images   (4 bytes)
        num rows     (4 bytes)
        num cols     (4 bytes)
        pixel data   (num images * rows * cols bytes, uint8)

    IDX label file header (big-endian):
        magic number (4 bytes) = 2049
        num labels   (4 bytes)
        label data   (num labels bytes, uint8)

Caches the parsed arrays as .npy so re-running doesn't re-download or re-parse.
"""

import gzip
import os
import struct
import urllib.request

import numpy as np
import torch
from torch.utils.data import Dataset

# Mirrors that serve the original 4 MNIST files. Tried in order until one works.
MIRRORS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "http://yann.lecun.com/exdb/mnist/",
]

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _download(filename, dest_path):
    if os.path.exists(dest_path):
        return
    last_err = None
    for mirror in MIRRORS:
        url = mirror + filename
        try:
            print(f"Downloading {url} ...")
            urllib.request.urlretrieve(url, dest_path)
            return
        except Exception as e:  # try next mirror
            last_err = e
            print(f"  failed ({e}), trying next mirror...")
    raise RuntimeError(f"Could not download {filename} from any mirror: {last_err}")


def _parse_idx_images(gz_path):
    with gzip.open(gz_path, "rb") as f:
        magic, num_images, num_rows, num_cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Bad magic number for image file: {magic}")
        buf = f.read(num_images * num_rows * num_cols)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, num_rows, num_cols)
    return data


def _parse_idx_labels(gz_path):
    with gzip.open(gz_path, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Bad magic number for label file: {magic}")
        buf = f.read(num_labels)
        labels = np.frombuffer(buf, dtype=np.uint8)
    return labels


def _load_split(root, train):
    os.makedirs(root, exist_ok=True)
    prefix = "train" if train else "test"
    img_key, lbl_key = (f"{prefix}_images", f"{prefix}_labels")

    images_npy = os.path.join(root, f"{prefix}_images.npy")
    labels_npy = os.path.join(root, f"{prefix}_labels.npy")

    if os.path.exists(images_npy) and os.path.exists(labels_npy):
        images = np.load(images_npy)
        labels = np.load(labels_npy)
        return images, labels

    img_gz = os.path.join(root, FILES[img_key])
    lbl_gz = os.path.join(root, FILES[lbl_key])
    _download(FILES[img_key], img_gz)
    _download(FILES[lbl_key], lbl_gz)

    images = _parse_idx_images(img_gz)
    labels = _parse_idx_labels(lbl_gz)

    np.save(images_npy, images)
    np.save(labels_npy, labels)
    return images, labels


class MNISTRaw(Dataset):
    """
    Drop-in replacement for torchvision.datasets.MNIST, parsed by hand.

    Returns (image, label) where image is a float32 tensor of shape (1, 28, 28)
    with pixel values scaled to [0, 1], matching transforms.ToTensor()'s behavior.
    """

    def __init__(self, root="./data", train=True):
        self.images, self.labels = _load_split(root, train)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].astype(np.float32) / 255.0  # (28, 28) in [0, 1]
        img = torch.from_numpy(img).unsqueeze(0)            # (1, 28, 28)
        label = int(self.labels[idx])
        return img, label


if __name__ == "__main__":
    # quick smoke test
    train_ds = MNISTRaw(root="./data", train=True)
    test_ds = MNISTRaw(root="./data", train=False)
    print(f"train: {len(train_ds)} samples, test: {len(test_ds)} samples")
    x, y = train_ds[0]
    print(f"sample image shape: {tuple(x.shape)}, dtype: {x.dtype}, label: {y}")