import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from models.siamese import get_siamese_model
from train import get_batch


if __name__ == "__main__":
    shape = (614, 421, 3)
    model = get_siamese_model(shape)
    model.load_weights('saved_models/saved-model-19.h5')
    files = os.listdir('./cards/')
    for i, image in enumerate(get_batch(os.listdir('./cards/'), 1, shape)):
        max_score = 0
        test_image = image[0]
        best_match = None
        start = datetime.now()
        for j, image2 in enumerate(get_batch(os.listdir('./cards/'), 1, shape)):
            score = model.predict([image, image2])
            if max_score < score:
                best_match = image2
                max_score = score
            if j > i + 10:
                break
        print(datetime.now() - start)
        print(max_score)
        fig = plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(np.squeeze(image))
        plt.title("Test Image")
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(np.squeeze(best_match))
        plt.title("Best Match")
        plt.axis('off')
        plt.show()


