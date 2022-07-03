# import cv2
# import numpy as np
# import streamlit as st
#
#
# def img2bright(photo):
#     # Read Image
#     img = photo
#     original = img.copy()
#
#     # convert image to HSV color space
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     hsv = np.array(hsv, dtype=np.float64)
#
#     # scale pixel values up for channel 1
#     hsv[:, :, 1] = hsv[:, :, 1] * 1.25
#     hsv[:, :, 1][hsv[:, :, 1] > 255] = 255
#
#     # scale pixel values up for channel 2
#     hsv[:, :, 2] = hsv[:, :, 2] * 1.25
#     hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
#     hsv = np.array(hsv, dtype=np.uint8)
#
#     # converting back to BGR used by OpenCV
#     img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
#     return img
#
#
# def img2enh(photo):
#     # Read Image
#     img = photo
#
#     dst = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
#     # sigma_s controls how much the image is smoothed - the larger its value,
#     # the more smoothed the image gets, but it's also slower to compute.
#     # sigma_r is important if you want to preserve edges while smoothing the image.
#     # Small sigma_r results in only very similar colors to be averaged (i.e. smoothed), while colors that differ much will stay intact.
#     kernel_sharpening = np.array([[-1, -1, -1],
#                                   [-1, 9, -1],
#                                   [-1, -1, -1]])
#     dst2 = cv2.filter2D(img, -1, kernel_sharpening)
#
#     return dst, dst2
#
#
# def img2inv(photo):
#     # Read Image
#     img = photo
#
#     res = cv2.bitwise_not(img)
#
#     return res
#
#
# def gamma_function1(channel, gamma):
#     invGamma = 1 / gamma
#     table = np.array([((i / 255.0) ** invGamma) * 255
#                       for i in np.arange(0, 256)]).astype("uint8")
#     channel = cv2.LUT(channel, table)
#     return channel
#
#
# def img2sum(photo):
#     # Read Image
#     img = photo
#
#     original = img.copy()
#     img[:, :, 0] = gamma_function1(img[:, :, 0], 1.25)
#     img[:, :, 2] = gamma_function1(img[:, :, 2], 0.75)
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     hsv[:, :, 1] = gamma_function1(hsv[:, :, 1], 0.8)
#     img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
#
#     return img
#
#
# def gamma_function2(channel, gamma):
#     invGamma = 1 / gamma
#     table = np.array([((i / 255.0) ** invGamma) * 255
#                       for i in np.arange(0, 256)]).astype("uint8")  # creating lookup table
#     channel = cv2.LUT(channel, table)
#     return channel
#
#
# def img2win(photo):
#     # Read Image
#     img = photo
#
#     original = img.copy()
#     img[:, :, 0] = gamma_function2(img[:, :, 0], 0.75)  # down scaling red channel
#     img[:, :, 2] = gamma_function2(img[:, :, 2], 1.25)  # up scaling blue channel
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     hsv[:, :, 1] = gamma_function2(hsv[:, :, 1], 1.2)  # up scaling saturation channel
#     img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
#
#     return img
#
#
# def img2sepia(photo):
#     # Read Image
#     img = photo
#
#     # converting to float to prevent loss
#     img = np.array(img, dtype=np.float64)
#
#     # multipying image with special sepia matrix
#     img = cv2.transform(img, np.matrix([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]]))
#
#     # normalizing values greater than 255 to 255
#     img[np.where(img > 255)] = 255
#
#     # converting back to int
#     img = np.array(img, dtype=np.uint8)
#
#     return img
#
#
# def hsv(img, l, u):
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     lower = np.array([l, 50, 70])  # setting lower HSV value
#     upper = np.array([u, 255, 255])  # setting upper HSV value
#     mask = cv2.inRange(hsv, lower, upper)  # generating mask
#     return mask
#
#
# def img2splash(photo):
#     # Read Image
#     img = photo
#     original = img.copy()
#
#     res = np.zeros(img.shape, np.uint8)  # creating blank mask for result
#     l = 25  # the lower range of Hue we want
#     u = 35  # the upper range of Hue we want
#     mask = hsv(img, l, u)
#     inv_mask = cv2.bitwise_not(mask)  # inverting mask
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     res1 = cv2.bitwise_and(img, img, mask=mask)  # region which has to be in color
#     res2 = cv2.bitwise_and(gray, gray, mask=inv_mask)  # region which has to be in grayscale
#     for i in range(3):
#         res[:, :, i] = res2  # storing grayscale mask to all three slices
#     img = cv2.bitwise_or(res1, res)  # joining grayscale and color region
#
#     return img
#
#
# def img2cont(photo):
#     img = photo
#     original = img.copy()
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     xp = [0, 64, 112, 128, 144, 192, 255]  # setting reference values
#     fp = [0, 16, 64, 128, 192, 240, 255]  # setting values to be taken for reference values
#     x = np.arange(256)
#     table = np.interp(x, xp, fp).astype('uint8')  # creating lookup table
#     img = cv2.LUT(gray, table)  # changing values based on lookup table
#
#     return img
#
#
# def img2emb(photo):
#     # Read Image
#     img = photo
#
#     # Storing the size of image in h and w
#     height, width = img.shape[:2]
#     y = np.ones((height, width), np.uint8) * 128
#     output = np.zeros((height, width), np.uint8)
#     # generating the kernels
#     # kernel for embossing bottom left side
#     kernel1 = np.array([[0, -1, -1],
#                         [1, 0, -1],
#                         [1, 1, 0]])
#     # kernel for embossing bottom right side
#     kernel2 = np.array([[-1, -1, 0],
#                         [-1, 0, 1],
#                         [0, 1, 1]])
#     # you can generate kernels for embossing top as well
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # emboss on bottom left side
#     output1 = cv2.add(cv2.filter2D(gray, -1, kernel1), y)
#     # emboss on bottom right side
#     output2 = cv2.add(cv2.filter2D(gray, -1, kernel2), y)
#     for i in range(height):
#         for j in range(width):
#             output[i, j] = max(output1[i, j], output2[i, j])  # combining both embosses to produce stronger emboss
#
#     return output
#
#
# def tv_60(photo):
#     # Read Image
#     img = photo
#
#     while True:
#         height, width = img.shape[:2]
#
#         val = st.slider('Noise Value', 0, 255, 128, key="na_lower")
#         thresh = st.slider('Threshold', 0, 100, 50, key="na_up")
#
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         for i in range(height):
#             for j in range(width):
#                 if np.random.randint(100) <= thresh:
#                     if np.random.randint(2) == 0:
#                         gray[i, j] = min(gray[i, j] + np.random.randint(0, val + 1),
#                                          255)  # adding noise to image and setting values > 255 to 255.
#                     else:
#                         gray[i, j] = max(gray[i, j] - np.random.randint(0, val + 1),
#                                          0)  # subtracting noise to image and setting values < 0 to 0.
#
#         st.image(gray, caption=f"Image with 60s TV Filter", use_column_width=True)
#
#
# def img2cartoon(photo):
#     # Read Image
#     img = photo
#     # Convert to Grey Image
#     grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     grey_img = cv2.medianBlur(grey_img, 5)
#     edges = cv2.adaptiveThreshold(grey_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
#     color = cv2.bilateralFilter(img, 9, 250, 250)
#     cartoon = cv2.bitwise_and(color, color, mask=edges)
#
#     # textimage=cv2.divide(grey_img,invblur_img, scale=256.0)
#     # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
#     # textimage = clahe.apply(textimage)
#
#     return cartoon
#
#
# def img2sketch(photo, k_size):
#     # Read Image
#     img = photo
#     # Convert to Grey Image
#     grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     # Invert Image
#     invert_img = cv2.bitwise_not(grey_img)
#     # invert_img=255-grey_img
#
#     # Blur image
#     blur_img = cv2.GaussianBlur(img, (k_size, k_size), 0)
#
#     # Invert Blurred Image
#     invblur_img = cv2.bitwise_not(blur_img)
#     # invblur_img=255-blur_img
#
#     # Sketch Image
#     sketch_img = cv2.divide(grey_img, invblur_img, scale=256.0)
#     # imporve contrast using histogram equilization
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
#     sketch_img = clahe.apply(sketch_img)
#
#     return sketch_img
#
#
# def exponential_function(channel, exp):
#     table = np.array([min((i ** exp), 255) for i in np.arange(0, 256)]).astype("uint8")  # creating table for exponent
#     channel = cv2.LUT(channel, table)
#     return channel
#
#
# def img2tone(img, number):
#     for i in range(3):
#         if i == number:
#             img[:, :, i] = exponential_function(img[:, :, i], 1.05)  # applying exponential function on slice
#         else:
#             img[:, :, i] = 0  # setting values of all other slices to 0
#     return img
#
#
# selected_box = st.sidebar.selectbox('Choose one of the filters', (
# 'None', 'Bright', 'Detail Enchance', 'Invert', 'Summer', 'Winter', 'Daylight', 'High Contrast', 'Sepia', 'Splash',
# 'Emboss', '60s TV', 'Dual Tone', 'Cartoon', 'Pencil Drawing', 'Comic'))
#
#
# def img2day(photo):
#     img = photo
#     image_HLS = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)  # Conversion to HLS
#     image_HLS = np.array(image_HLS, dtype=np.float64)
#     daylight = 1.15
#     image_HLS[:, :, 1] = image_HLS[:, :, 1] * daylight  # scale pixel values up for channel 1(Lightness)
#     image_HLS[:, :, 1][image_HLS[:, :, 1] > 255] = 255  # Sets all values above 255 to 255
#     image_HLS = np.array(image_HLS, dtype=np.uint8)
#     image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2BGR)  # Conversion to RGB
#
#     return image_RGB
#
#
# def img2pen(photo):
#     img = photo
#     dst_gray, dst_color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07,
#                                            shade_factor=0.05)  # inbuilt function to generate pencil sketch in both color and grayscale
#     # sigma_s controls the size of the neighborhood. Range 1 - 200
#     # sigma_r controls the how dissimilar colors within the neighborhood will be averaged. A larger sigma_r results in large regions of constant color. Range 0 - 1
#     # shade_factor is a simple scaling of the output image intensity. The higher the value, the brighter is the result. Range 0 - 0.1
#     return dst_color
#
#
# def comic(photo):
#     # Read Image
#     img = photo
#     cpy = img.reshape((-1, 3))
#     cpy = np.float32(cpy)
#
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#
#     while True:
#         k = st.slider('No. of Clusters (k)', 10, 30, 1, key="na_lower")
#
#         ret, label, center = cv2.kmeans(cpy, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
#         center = np.uint8(center)
#         res = center[label.flatten()]
#         res2 = res.reshape((img.shape))
#
#         st.image(res2, caption=f"Image with Comic Filter", use_column_width=True)
