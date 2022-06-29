import numpy as np
import cv2

# 0 pixels are the "null" pixels, -1 pixels are background pixels, 1 pixels are foreground pixels
border_kernel = np.array([[0, -1, -1], [1, 1, -1], [0, 1, 0]], dtype="int")
corner_kernel = np.array([[-1, -1, -1], [0, 1, 0], [1, 1, 1]], dtype="int")

kernels = [
    border_kernel,
    corner_kernel,
    np.rot90(border_kernel),
    np.rot90(corner_kernel),
    np.rot90(np.rot90(border_kernel)),
    np.rot90(np.rot90(corner_kernel)),
    np.rot90(np.rot90(np.rot90(border_kernel))),
    np.rot90(np.rot90(np.rot90(corner_kernel)))
]

spur1_kernel = np.array([[-1, -1, -1], [-1, 1, -1], [0, 0, -1]], dtype="int")
spur2_kernel = np.array([[-1, -1, -1], [-1, 1, -1], [-1, 0, 0]], dtype="int")

spur_kernels = [
    spur1_kernel,
    spur2_kernel,
    np.rot90(spur1_kernel),
    np.rot90(spur2_kernel),
    np.rot90(np.rot90(spur1_kernel)),
    np.rot90(np.rot90(spur2_kernel)),
    np.rot90(np.rot90(np.rot90(spur1_kernel))),
    np.rot90(np.rot90(np.rot90(spur2_kernel))),
]


def morph_thinning_skeletonize(image, pruning, pruning_iterations):
    cleaned_skeleton = np.pad(
        image, pad_width=1, mode='constant', constant_values=0)
    cleaned_skeleton = cleaned_skeleton.astype(np.uint8)

    skeleton = np.zeros(cleaned_skeleton.shape, dtype=np.uint8)
    skeleton[:, :] = cleaned_skeleton > 0
    pixel_removed = True

    while pixel_removed:
        pixel_removed = False

        cleaned_skeleton = thin(cleaned_skeleton, kernels)

        diff = np.zeros(skeleton.shape, dtype=np.uint8)
        cv2.absdiff(cleaned_skeleton, skeleton, diff)
        if cv2.countNonZero(diff) > 0:
            skeleton = cleaned_skeleton
            pixel_removed = True

    if pruning:
        pixel_removed = True
        for i in range(pruning_iterations):
            if not pixel_removed:
                break

            cleaned_skeleton = thin(cleaned_skeleton, spur_kernels)

            diff = np.zeros(skeleton.shape, dtype=np.uint8)
            cv2.absdiff(skeleton, cleaned_skeleton, diff)
            if cv2.countNonZero(diff) > 0:
                skeleton = cleaned_skeleton
                pixel_removed = True

    return skeleton[1:-1, 1:-1]


def thin(image, kernels):
    for kernel in kernels:
        out_image = cv2.morphologyEx(image, cv2.MORPH_HITMISS, kernel)
        image = image - out_image
    return image
