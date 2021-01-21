import os
import random

import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

from models.siamese import get_siamese_model
import cv2


def get_batch(files: list, batch_size: int, shape: tuple):
    batch = []
    for file in files:
        image = cv2.imread(f'./cards/{file}')
        if image.shape != shape:
            image = cv2.resize(image, (shape[1], shape[0]))
        batch.append(image)
        if len(batch) % batch_size == 0:
            yield np.stack(batch)
            batch = []


def generate_images(batch_size: int, shape: tuple):
    """Create batch of n pairs, half same class, half different class"""
    files = os.listdir('./cards/')
    while True:
        random.shuffle(files)
        for batch in get_batch(files, batch_size, shape):
            w, h = shape[0], shape[1]
            pairs = [np.zeros((batch_size, w, h, 3)) for _ in range(2)]
            targets = np.zeros((batch_size,))
            targets[batch_size // 2:] = 1
            for i in range(batch_size):
                category = i
                pairs[0][i, :, :, :] = batch[category, ]
                # pick images of same class for 1st half, different for 2nd
                category_2 = category if i >= batch_size // 2 else (category + random.randrange(1, batch_size)) % batch_size
                pairs[1][i, :, :, :] = batch[category_2, ]
            yield pairs, targets


if __name__ == "__main__":
    shape = (614, 421, 3)
    epochs = 20
    batch_size = 16
    model = get_siamese_model(shape)
    optimizer = Adam(lr=0.00006)
    filepath = "saved_models/saved-model-{epoch:02d}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_weights_only=True)
    model.compile(loss="binary_crossentropy", optimizer=optimizer)
    model.fit_generator(generator=generate_images(batch_size, shape),
                        epochs=epochs,
                        steps_per_epoch=len(os.listdir('./cards/')) / batch_size,
                        use_multiprocessing=True,
                        callbacks=[checkpoint],
                        workers=os.cpu_count(),
                        shuffle=True)
