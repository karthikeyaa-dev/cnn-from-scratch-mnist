import numpy as np
import gzip
import os
import random

'''def _open_file(filename):
    if filename.endswith(".gz"):
        return gzip.open(filename, "rb")
    return open(filename, "rb")

def load_mnist_images(filename):
    with _open_file(filename) as f:
        data = f.read()
    images = np.frombuffer(data, dtype=np.uint8, offset=16)
    return images.reshape(-1, 28, 28)

def load_mnist_labels(filename):
    with _open_file(filename) as f:
        data = f.read()
    labels = np.frombuffer(data, dtype=np.uint8, offset=8)
    return labels'''

def default_collate(batch):
    """
    batch = [(x1, y1), (x2, y2), ..., (xB, yB)]
    """
    xs, ys = zip(*batch)          # unzip
    X = torch.stack(xs)           # [B, ...]
    Y = torch.tensor(ys)          # [B]
    return X, Y


class LazyMNISTDataset:

    def __init__(self, images_file, labels_file, normalize=True, add_channel=True, flatten=False):
        self.images_file = images_file
        self.labels_file = labels_file
        self.normalize = normalize
        self.add_channel = add_channel
        self.flatten = flatten

        # Load only labels into memory (small)
        with _open_file(labels_file) as f:
            data = f.read()
        self.labels = np.frombuffer(data, dtype=np.uint8, offset=8).astype(np.int64)

        # Number of images
        with _open_file(images_file) as f:
            data = f.read()
        num_images = (len(data) - 16) // (28 * 28)
        self.num_images = num_images

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        # Lazy load a single image
        with _open_file(self.images_file) as f:
            f.seek(16 + idx * 28 * 28)  # Skip header + previous images
            image_data = f.read(28 * 28)
        image = np.frombuffer(image_data, dtype=np.uint8).astype(np.float32)

        if self.normalize:
            image /= 255.0
        if self.add_channel:
            image = image[..., np.newaxis]  # (28,28,1)
        if self.flatten:
            image = image.reshape(-1)  # (784,)

        label = self.labels[idx]
        return torch.tensor(image), torch.tensor(label)


class MNISTDataLoader:

    def __init__(self, dataset, batch_size=32, shuffle=True, collate_fn=None, drop_last=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn or default_collate
        self.drop_last = drop_last or False

    def __iter__(self):
        # 1. Create indices
        indices = list(range(len(self.dataset)))

        # 2. Shuffle indices if needed
        if self.shuffle:
            random.shuffle(indices)

        # 3. Accumulate samples into batches
        batch = []
        for idx in indices:
            sample = self.dataset[idx]
            batch.append(sample)

            if len(batch) == self.batch_size:
                yield self.collate_fn(batch)
                batch = []

        # 4. Yield last batch (if any)
        if len(batch) > 0 and not self.drop_last:
            yield self.collate_fn(batch)

    def __len__(self):
        n = len(self.dataset) // self.batch_size
        if not self.drop_last and len(self.dataset) % self.batch_size != 0:
            n += 1
        return n

