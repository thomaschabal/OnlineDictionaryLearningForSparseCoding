import cv2
import matplotlib.pyplot as plt
import numpy as np


def add_text_on_image(img, texts=[], scale=1, bottom_left=(10, 200), color=(255, 0, 0), line_type=2):
    h, w = img.shape[:2]
    ratio = max(200 / h, 200 / w)
    if ratio > 1:
        img = cv2.resize(img, (int(ratio * h), int(ratio * w)))

    for idx, text in enumerate(texts):
        cv2.putText(img, text,
                    (bottom_left[0], bottom_left[1] + 25 * idx),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    scale,
                    color,
                    line_type)

    if ratio > 1:
        img = cv2.resize(img, (h, w))
    return img


def show_imgs(imgs, titles, figsize=(15, 15), suptitle=None):
    fig, axes = plt.subplots(ncols=len(imgs), figsize=figsize)
    if len(imgs) == 1:
        axes.imshow(imgs[0])
        axes.set_title(titles[0])
    else:
        for idx, img in enumerate(imgs):
            axes[idx].imshow(img)
            axes[idx].set_title(titles[idx])
    if suptitle is not None:
        fig.suptitle(suptitle)
    plt.show()


# ===== Modify images to shift color channels and evaluate robustness of the dictionary learning =====

def attenuate_color_channel_images(X: np.ndarray, channel: int):
    """
        X: shape (n_samples, image_height, image_width, channels)
        channel: int in {0, 1, 2} for the RGB colors, channel to keep identical, while attenuating the two others
    """
    for c in range(3):
        if channel != c:
            X[:, :, :, c] *= 0.25
    return X


def attenuate_color_channel_features(X: np.ndarray, channel: int, img_size: tuple):
    """
        X: shape (n_samples, feature_dimension)
        channel: int in {0, 1, 2} for the RGB colors, channel to keep identical, while attenuating the two others
        img_size: tuple (height, width)
    """
    if type(img_size) == int:
        img_size = (img_size, img_size)

    X_images = np.reshape(X, (X.shape[0], img_size[0], img_size[1], 3))
    X_attenutated_images = attenuate_color_channel_images(X_images, channel)
    X_attenuated = np.reshape(X_attenutated_images, X.shape)

    return X_attenuated
