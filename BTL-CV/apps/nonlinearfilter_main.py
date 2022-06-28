import this

import streamlit as st
from PIL import Image
import cv2
import numpy as np
from skimage.util import random_noise
from PIL import Image, ImageFilter
import copy
def app():
    selected_box = st.sidebar.selectbox('Choose one filter',
                                        ('None', 'Median', 'Bilateral', 'Entropy','Max/Min'))

    if selected_box == 'None':
        st.title('Non-Linear Filter')
        ## Add bulletins
        st.subheader("Select from the following Non-linear Filter", anchor=None)
        st.header("CV", anchor=None)

        st.subheader("Available Non-linear Filter", anchor=None)

        st.markdown(
            '<ul> <li> Median <li> Bilateral <li> Entropy  </ul>',
            unsafe_allow_html=True)
    # Begin upload img

    def load_image():
        img_file_buffer = st.file_uploader("Upload an image", type=["jpg", "jpeg", 'png'])
        if img_file_buffer is not None:
            image = np.array(Image.open(img_file_buffer))
            st.image(image, caption=f"Original Image", use_column_width=True)
            return image
        else:
            return None

    # add noise to image
    def add_noise(original_image, mode="s&p"):
        noise_img = random_noise(original_image, mode)
        # print(noise_img)
        noise_img = np.array(255 * noise_img, dtype="uint8")
        return noise_img
    # def noisy(image,noise_typ):
    #     if noise_typ == "gauss":
    #         row, col, ch = image.shape
    #         mean = 0
    #         var = 0.1
    #         sigma = var ** 0.5
    #         gauss = np.random.normal(mean, sigma, (row, col, ch))
    #         gauss = gauss.reshape(row, col, ch)
    #         noisy = image + gauss
    #         return noisy
    #     elif noise_typ == "s&p":
    #         row, col, ch = image.shape
    #         s_vs_p = 0.5
    #         amount = 0.004
    #         out = np.copy(image)
    #         # Salt mode
    #         num_salt = np.ceil(amount * image.size * s_vs_p)
    #         coords = [np.random.randint(0, i - 1, int(num_salt))
    #                   for i in image.shape]
    #         out[coords] = 1
    #
    #         # Pepper mode
    #         num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    #         coords = [np.random.randint(0, i - 1, int(num_pepper))
    #                   for i in image.shape]
    #         out[coords] = 0
    #         return out
    #     elif noise_typ == "poisson":
    #         vals = len(np.unique(image))
    #         vals = 2 ** np.ceil(np.log2(vals))
    #         noisy = np.random.poisson(image * vals) / float(vals)
    #         return noisy
    #     elif noise_typ == "speckle":
    #         row, col, ch = image.shape
    #         gauss = np.random.randn(row, col, ch)
    #         gauss = gauss.reshape(row, col, ch)
    #         noisy = image + image * gauss
    #         return noisy

    # Median filter

    def median_filter(original_image, level):
        output_img = cv2.medianBlur(original_image, level);
        return output_img

    if selected_box == 'Median':
        noise_mode = st.sidebar.selectbox('Choose noise mode', ('gaussian', 's&p', 'poisson', 'speckle'))
        median_level = st.sidebar.selectbox('Choose one of the level', (1, 3, 5, 7))
        st.title('Median filter')
        image = load_image()
        convert_btn = st.button('ADD NOISE AND CONVERT')
        col1, col2 = st.columns(2)
        if convert_btn:
            noise_image = add_noise(image, noise_mode)
            output_image = median_filter(noise_image, median_level)
            st.image(noise_image, caption=f"Image with Noise", use_column_width=True, clamp=True)
            st.image(output_image, caption=f"Output image", use_column_width=True, clamp=True)

    #   Bilateral filter: làm mịn ảnh
    def bilateral_filter(original_image, radius, sigmaColor = 75 , sigmaSpace=75):
        # bilateral_image = cv2.bilateralFilter(args[0], args[1], st.session_state.color, st.session_state.space)
        bilateral_image = cv2.bilateralFilter(original_image, radius, sigmaColor, sigmaSpace)
        return bilateral_image
    if selected_box == "Bilateral":
        image = load_image()
        radius_level = st.sidebar.selectbox('Choose radius ', (1, 2, 3, 4, 5))
        # sigmaColor = st.sidebar.slider("Color: ", 0, 200, 75, 5, key="color", on_change=bilateral_filter, args=(image, radius_level))
        # sigmaSpace = st.sidebar.slider("Space: ", 0, 200, 75, 5, key="space", on_change=bilateral_filter, args=(image, radius_level))
        color = st.sidebar.slider("Color: ", 0, 200, 75, 5, key="color")
        space = st.sidebar.slider("Space: ", 0, 200, 75, 5, key="space")
        st.title('Bilateral filter')
        convert_btn = st.button('CONVERT')
        if convert_btn:
            output_image = bilateral_filter(image, radius_level * 10, color, space)
            st.image(output_image, caption=f"Output image", use_column_width=True, clamp=True)

    #   Max/Min filter
    def max_min_filtering(original_image, mode , kernel_size):
        if mode == "Max":
            output_image = original_image.filter(ImageFilter.MaxFilter(kernel_size))
        if mode == "Min":
            output_image = original_image.filter(ImageFilter.MaxFilter(kernel_size))
        return output_image

    if selected_box == "Max/Min":
        img_file_buffer = st.file_uploader("Upload an image", type=["jpg", "jpeg", 'png'])
        if img_file_buffer is not None:
            image = Image.open(img_file_buffer)
            st.image(image, caption=f"Original Image", use_column_width=True)
            mode = st.sidebar.selectbox("Choose mode", ("Max", "Min"))
            filter_size = st.sidebar.slider("Choose filter size: ", 1, 5, 3, 2, key="filter_size")
            st.title('Max/Min filter')
            convert_btn = st.button('CONVERT')
            if convert_btn:
                output_image = max_min_filtering(image, mode , filter_size)
                st.image(output_image, caption=f"Output image", use_column_width=True, clamp=True)





