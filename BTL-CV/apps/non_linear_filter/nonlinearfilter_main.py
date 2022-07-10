import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageFilter
from scipy.signal import convolve2d
from skimage.util import random_noise


def app():
    selected_box = st.sidebar.selectbox('Choose one filter',
                                        ('None', 'Median', 'Bilateral', 'Max/Min', 'Kuwahara'))

    if selected_box == 'None':
        st.title('Non-Linear Filter')
        ## Add bulletins
        st.subheader("Select from the following Non-linear Filter", anchor=None)
        st.header("CV", anchor=None)

        st.subheader("Available Non-linear Filter", anchor=None)

        st.markdown(
            '<ul> <li> Median <li> Bilateral <li> Max/Min <li> Kuwahara </ul>',
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
    def bilateral_filter(original_image, radius, sigmaColor=75, sigmaSpace=75):
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
    def max_min_filtering(original_image, mode, kernel_size):
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
                output_image = max_min_filtering(image, mode, filter_size)
                st.image(output_image, caption=f"Output image", use_column_width=True, clamp=True)

    #   Kuwahara filter
    def kuwahara_filter(original, winsize):

        image = original.astype(np.float64)
        # make sure window size is correct
        if winsize % 4 != 1:
            raise Exception("Invalid winsize %s: winsize must follow formula: w = 4*n+1." % winsize)

        # Build subwindows
        tmpAvgKerRow = np.hstack((np.ones((1, (winsize - 1) // 2 + 1)), np.zeros((1, (winsize - 1) // 2))))
        tmpPadder = np.zeros((1, winsize))
        tmpavgker = np.tile(tmpAvgKerRow, ((winsize - 1) // 2 + 1, 1))
        tmpavgker = np.vstack((tmpavgker, np.tile(tmpPadder, ((winsize - 1) // 2, 1))))
        tmpavgker = tmpavgker / np.sum(tmpavgker)

        # tmpavgker is a 'north-west' subwindow (marked as 'a' above) #
        # we build a vector of convolution kernels for computing average and
        # variance
        avgker = np.empty((4, winsize, winsize))  # make an empty vector of arrays
        avgker[0] = tmpavgker  # North-west (a)
        avgker[1] = np.fliplr(tmpavgker)  # North-east (b)
        avgker[2] = np.flipud(tmpavgker)  # South-west (c)
        avgker[3] = np.fliplr(avgker[2])  # South-east (d)

        # Create a pixel-by-pixel square of the image
        squaredImg = image ** 2

        # preallocate these arrays to make it apparently %15 faster
        avgs = np.zeros([4, image.shape[0], image.shape[1]])
        stddevs = avgs.copy()

        # Calculation of averages and variances on subwindows
        for k in range(4):
            # mean on subwindow
            avgs[k] = convolve2d(image, avgker[k], mode='same')
            # mean of squares on subwindow
            stddevs[k] = convolve2d(squaredImg, avgker[k], mode='same')
            # variance on subwindow
            stddevs[k] = stddevs[k] - avgs[k] ** 2
        # Choice of index with minimum variance
        indices = np.argmin(stddevs, 0)  # returns index of subwindow with smallest variance

        # Building the filtered image (with nested for loops)
        filtered = np.zeros(original.shape)
        for row in range(original.shape[0]):
            for col in range(original.shape[1]):
                filtered[row, col] = avgs[indices[row, col], row, col]

        # filtered=filtered.astype(np.uint8)
        return filtered.astype(np.uint8)

    if selected_box == "Kuwahara":
        image = load_image()
        win_size = st.sidebar.selectbox("Choose filter size:", (5, 9, 25))
        st.title('Kuwahara filter')
        convert_btn = st.button('CONVERT')
        if convert_btn:
            output_image = kuwahara_filter(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY), win_size)
            st.image(output_image, caption=f"Output image", use_column_width=True, clamp=True)
