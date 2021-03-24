import numpy as np


def infer_img(img, dict_learner):
    h, w = img.shape[:2]
    if len(img.shape) == 2:
        img = img.flatten()[np.newaxis, :]
    img_transformed = dict_learner.transform(img)
    img_hat = img_transformed @ dict_learner.components_
    img_hat = np.reshape(img_hat, (1, h, w))[0]
    return img_hat
