import cv2
import matplotlib.pyplot as plt


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


def show_imgs(imgs, titles):
    _, axes = plt.subplots(ncols=len(imgs), figsize=(15, 15))
    for idx, img in enumerate(imgs):
        axes[idx].imshow(img)
        axes[idx].set_title(titles[idx])
    plt.show()
