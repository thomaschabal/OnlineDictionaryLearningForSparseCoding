import numpy as np
from .utils import add_text_on_image, show_imgs
from .metrics import l2_images_error


def infer_img(img, dict_learner):
    h, w = img.shape[:2]
    if len(img.shape) == 2:
        img = img.flatten()[np.newaxis, :]
    img_transformed = dict_learner.transform(img)
    img_hat = img_transformed @ dict_learner.components_
    img_hat = np.reshape(img_hat, (1, h, w))[0]
    return img_hat


def forge_and_reconstruct(img, dict_learner, text, bottom_left=(10, 30), color=(1, 0, 0), display=True):
    forged_img = add_text_on_image(
        img, text, scale=0.8, line_type=2, bottom_left=bottom_left, color=color)

    img_hat = infer_img(img, dict_learner)
    forged_img_hat = infer_img(forged_img, dict_learner)

    if display:
        l2_original_forged = l2_images_error(img, forged_img)
        title_originals = f"L2 error between original and forged images: {l2_original_forged:.3f}\n"

        l2_reconstructions = l2_images_error(img_hat, forged_img_hat)
        title_reconstructs = f"L2 error between respective reconstructions: {l2_reconstructions:.3f}"

        suptitle = title_originals + title_reconstructs
        show_imgs([img, forged_img, img_hat, forged_img_hat], ["Original", "Forged",
                  "Original reconstructed", "Forged reconstructed"],
                  suptitle=suptitle, figsize=(15, 4))

    return forged_img, img_hat, forged_img_hat
