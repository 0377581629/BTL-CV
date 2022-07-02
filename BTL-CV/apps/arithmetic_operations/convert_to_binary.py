import numpy as np
import cv2

def binary_pic(image):
    #convert to gray image
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #threshold
    (ret,image) = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    return image

