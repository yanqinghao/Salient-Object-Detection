import os
import numpy as np
import imageio
from PIL import Image


def composite4(fg, bg, a, w, h):
    fg = np.array(fg, np.float32)
    bg_h, bg_w = bg.shape[:2]
    x = 0
    if bg_w > w:
        x = np.random.randint(0, bg_w - w)
    y = 0
    if bg_h > h:
        y = np.random.randint(0, bg_h - h)
    bg = np.array(bg[y : y + h, x : x + w], np.float32)
    alpha = np.zeros((h, w, 1), np.float32)
    alpha[:, :, 0] = a / 255.0
    im = alpha * fg + (1 - alpha) * bg
    im = im.astype(np.uint8)
    return im, bg


def rgba2rgb(img):
    return img[:, :, :3] * np.expand_dims(img[:, :, 3], 2)


def get_all_files(dir):
    files_ = []
    list = [i for i in os.listdir(dir)]
    for i in range(0, len(list)):
        path = os.path.join(dir, list[i])
        if os.path.isdir(path):
            files_.extend(get_all_files(path))
        if not os.path.isdir(path):
            files_.append(path)
    return files_
