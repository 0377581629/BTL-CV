import numpy as np
import cv2


def simple_thresholding_thresh_binary(original_image, threshold):
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(gray_image, cv2.COLOR_BGR2RGB)

    (thresh, blackAndWhiteImage) = cv2.threshold(
        img, threshold, 255, cv2.THRESH_BINARY)
    return blackAndWhiteImage


def simple_thresholding_thresh_binary_inv(original_image, threshold):
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(gray_image, cv2.COLOR_BGR2RGB)

    (thresh, blackAndWhiteImage) = cv2.threshold(
        img, threshold, 255, cv2.THRESH_BINARY_INV)
    return blackAndWhiteImage


def otsu_thresholding(original_image):
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    (thresh, blackAndWhiteImage) = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return thresh, blackAndWhiteImage


def adaptive_mean_thresholding(original_image, ):
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    blackAndWhiteImage = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2)
    return blackAndWhiteImage


def adaptive_gaussian_thresholding(original_image, ):
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

    blackAndWhiteImage = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
    return blackAndWhiteImage
