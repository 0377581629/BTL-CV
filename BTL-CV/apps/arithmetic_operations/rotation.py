import cv2
import numpy as np
from PIL import Image
from scipy import ndimage


def warpaffine(image):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    rows, cols, ch = image.shape
    pts = np.float32([[50, 50],
                      [200, 50],
                      [50, 200]])
    pts2 = np.float32([[50, 100],
                       [200, 50],
                       [150, 200]])

    points = cv2.getAffineTransform(pts, pts2)
    img = cv2.warpAffine(image, points, (cols, rows))
    img_conv = Image.fromarray(img)
    return img_conv


def rotating(image, *argv):
    rotated = ndimage.rotate(image, *argv)
    rot_img = Image.fromarray(rotated)
    return rot_img


def flipping(image, *argv):
    flipped = cv2.flip(image, *argv)
    img_flipped = Image.fromarray(flipped)
    return img_flipped
