import numpy as np
import cv2
import matplotlib.pyplot as plt

# example adapted from opencv: https://docs.opencv.org/trunk/db/d06/tutorial_hitOrMiss.html
input_image = np.array((
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 255, 255, 255, 255, 0, 0, 255, 0, 0],
    [0, 0, 0, 255, 255, 255, 255, 0, 0, 0, 0, 0],
    [0, 0, 0, 255, 255, 255, 255, 255, 0, 0, 0, 0],
    [0, 0, 0, 0, 255, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 255, 0, 0, 0, 255, 255, 0, 0],
    [0, 0, 0, 255, 0, 255, 0, 0, 255, 255, 0, 0],
    [0, 0, 0, 255, 255, 255, 0, 0, 255, 255, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), dtype="uint8")

# 0 = don't care
# 1 = foreground
# -1 = background

kernel = np.array((
    [0, 1, 0],
    [1, -1, 1],
    [0, 1, 0]), dtype="int")

if __name__ == '__main__':
    output_image = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel, cv2.BORDER_CONSTANT)
    plt.figure(figsize=(15, 15))

    plt.subplot(1, 3, 1)
    plt.axis("off")
    plt.title("Original Image", fontsize=15)
    plt.imshow(input_image)
    plt.subplot(1, 3, 2)
    plt.axis("off")
    plt.title("Pattern", fontsize=15)
    plt.imshow(kernel)
    plt.subplot(1, 3, 3)
    plt.axis("off")
    plt.title("Output", fontsize=15)
    plt.imshow(output_image)
    plt.show()
