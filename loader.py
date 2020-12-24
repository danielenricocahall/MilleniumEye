import numpy as np
import numpy.random as rng


class SiameseLoader:
    """For loading batches and testing tasks to a siamese net"""

    def __init__(self, x_train):
        self.x_train = x_train
        self.n_classes, self.n_examples, self.w, self.h = x_train.shape

    def get_batch(self, batch_size, shape, x_train):
        """Create batch of n pairs, half same class, half different class"""
        n_classes = x_train.shape[0]
        w, h = shape[0], shape[1]
        categories = rng.choice(n_classes, size=(batch_size,), replace=False)
        pairs = [np.zeros((batch_size, w, h, 1)) for i in range(2)]
        targets = np.zeros((batch_size,))
        targets[batch_size // 2:] = 1
        for i in range(batch_size):
            category = categories[i]
            idx_1 = rng.randint(0, 1)
            pairs[0][i, :, :, :] = x_train[category, idx_1].reshape(w, h, 1)
            idx_2 = rng.randint(0, 1)
            # pick images of same class for 1st half, different for 2nd
            category_2 = category if i >= batch_size // 2 else (category + rng.randint(1, n_classes)) % n_classes
            pairs[1][i, :, :, :] = x_train[category_2, idx_2].reshape(w, h, 1)
        return pairs, targets

