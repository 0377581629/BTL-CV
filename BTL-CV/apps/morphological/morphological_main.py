import streamlit as st
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

from apps.morphological.convert_img_to_binary import simple_thresholding_thresh_binary, \
    simple_thresholding_thresh_binary_inv, otsu_thresholding, adaptive_mean_thresholding, adaptive_gaussian_thresholding
from apps.morphological.convex_hull import convex_hull
from apps.morphological.extract_components import extract_components
from apps.morphological.region_filling import region_filling
from apps.morphological.skeletonization import morph_thinning_skeletonize


def app():
    selected_box = st.sidebar.selectbox('Choose one of the operations',
                                        ('None', 'Convert to Binary', 'Erosion', 'Dilation', 'Opening', 'Closing',
                                         'Skeletonization',
                                         'Border Separation', 'Gradient', 'Region Filling',
                                         'Extract Components', 'Convex Hull', 'Top Hat', 'Black Hat'))

    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Begin upload img

    def load_image():
        img_file_buffer = st.file_uploader("Upload an image", type=["jpg", "jpeg", 'png'])
        if img_file_buffer is not None:
            image = np.array(Image.open(img_file_buffer))
            st.image(image, caption=f"Original Image", use_column_width=True)
            return image
        else:
            return None

    # End upload img

    if selected_box == 'None':
        st.title('Morphological')
        ## Add bulletins
        st.subheader("Select from the following morphological", anchor=None)
        st.header("CV", anchor=None)

        st.subheader("Available Morphological Operations", anchor=None)

        st.markdown(
            '<ul> <li> Convert to Binary <li> Erosion <li> Dilation <li> Opening <li> Closing <li> Skeletonization '
            '<li> Border Separation <li> Gradient <li> Region Filling <li> Extract '
            'Components <li> Convex Hull <li> Top Hat <li> Black Hat </ul>',
            unsafe_allow_html=True)

    # Begin Convert to Binary

    if selected_box == 'Convert to Binary':

        selected_box_type_of_convert = st.sidebar.selectbox('Choose one of the type',
                                                            ('Simple Thresholding', 'Simple Thresholding Invert',
                                                             'Otsu Thresholding', 'Adaptive Mean Thresholding',
                                                             'Adaptive Gaussian Thresholding'))

        if selected_box_type_of_convert == 'Simple Thresholding':
            selected_box_operation_level = st.sidebar.selectbox('Choose threshold',
                                                                (
                                                                    10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120,
                                                                    130,
                                                                    140,
                                                                    150, 160, 170, 180, 190, 200, 210, 220, 230, 240,
                                                                    250))
            st.title('Simple Thresholding')
            image = load_image()
            useWH = st.button('CONVERT')

            if useWH:
                resized_image = simple_thresholding_thresh_binary(image, selected_box_operation_level)
                st.image(resized_image, caption=f"Image with Simple Thresholding", use_column_width=True)

        if selected_box_type_of_convert == 'Simple Thresholding Invert':
            selected_box_operation_level = st.sidebar.selectbox('Choose threshold',
                                                                (
                                                                    10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120,
                                                                    130,
                                                                    140,
                                                                    150, 160, 170, 180, 190, 200, 210, 220, 230, 240,
                                                                    250))
            st.title('Simple Thresholding Invert')
            image = load_image()
            useWH = st.button('CONVERT')

            if useWH:
                resized_image = simple_thresholding_thresh_binary_inv(image, selected_box_operation_level)
                st.image(resized_image, caption=f"Image with Simple Thresholding Invert", use_column_width=True)

        if selected_box_type_of_convert == 'Otsu Thresholding':
            st.title('Otsu Thresholding')
            image = load_image()
            useWH = st.button('CONVERT')

            if useWH:
                thresh, resized_image = otsu_thresholding(image)
                st.image(resized_image, caption=f"Image with Otsu Thresholding with Threshold = " + str(thresh),
                         use_column_width=True)

        if selected_box_type_of_convert == 'Adaptive Mean Thresholding':
            st.title('Adaptive Mean Thresholding')
            image = load_image()
            useWH = st.button('CONVERT')

            if useWH:
                resized_image = adaptive_mean_thresholding(image)
                st.image(resized_image, caption=f"Image with Adaptive Mean Thresholding ", use_column_width=True)

        if selected_box_type_of_convert == 'Adaptive Gaussian Thresholding':
            st.title('Adaptive Gaussian Thresholding')
            image = load_image()
            useWH = st.button('CONVERT')

            if useWH:
                resized_image = adaptive_gaussian_thresholding(image)
                st.image(resized_image, caption=f"Image with Adaptive Gaussian Thresholding ", use_column_width=True)

    # End Convert to Binary

    # Begin Erosion

    def erosion(original_image, level):
        # Read Image
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(gray_image, cv2.COLOR_BGR2RGB)

        (thresh, blackAndWhiteImage) = cv2.threshold(
            img, 127, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)
        output_image = cv2.erode(blackAndWhiteImage, kernel, iterations=level)

        return output_image

    if selected_box == 'Erosion':

        selected_box_operation_level = st.sidebar.selectbox('Choose number of eroson', (1, 2, 3, 4, 5, 6, 7, 8, 9, 10))

        st.title('Erosion Morphological')
        image = load_image()
        useWH = st.button('CONVERT')

        if useWH:
            resized_image = erosion(image, selected_box_operation_level)
            st.image(resized_image, caption=f"Image with Erosion", use_column_width=True)

    # End Erosion

    # Begin Dilation

    def dilation(original_image, level):
        # Read Image
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(gray_image, cv2.COLOR_BGR2RGB)

        (thresh, blackAndWhiteImage) = cv2.threshold(
            img, 127, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)
        output_image = cv2.dilate(blackAndWhiteImage, kernel, iterations=level)

        return output_image

    if selected_box == 'Dilation':

        selected_box_operation_level = st.sidebar.selectbox('Choose number of Dilation',
                                                            (1, 2, 3, 4, 5, 6, 7, 8, 9, 10))

        st.title('Dilation Morphological')
        image = load_image()
        useWH = st.button('CONVERT')

        if useWH:
            resized_image = dilation(image, selected_box_operation_level)
            st.image(resized_image, caption=f"Image with Dilation", use_column_width=True)

    # End Dilation

    # Begin Opening

    def opening(original_image, level):
        # Read Image
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(gray_image, cv2.COLOR_BGR2RGB)

        binr = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)[1]

        kernel = np.ones((3, 3), np.uint8)
        output_image = cv2.morphologyEx(binr, cv2.MORPH_OPEN, kernel, iterations=level)

        return output_image

    if selected_box == 'Opening':

        selected_box_operation_level = st.sidebar.selectbox('Choose number of Opening', (1, 2, 3, 4, 5, 6, 7, 8, 9, 10))

        st.title('Opening Morphological')
        image = load_image()
        useWH = st.button('CONVERT')

        if useWH:
            resized_image = opening(image, selected_box_operation_level)
            st.image(resized_image, caption=f"Image with Opening", use_column_width=True)

    # End Opening

    # Begin Closing

    def closing(original_image, level):
        # Read Image
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(gray_image, cv2.COLOR_BGR2RGB)

        binr = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)[1]

        kernel = np.ones((3, 3), np.uint8)
        output_image = cv2.morphologyEx(binr, cv2.MORPH_CLOSE, kernel, iterations=level)

        return output_image

    if selected_box == 'Closing':

        selected_box_operation_level = st.sidebar.selectbox('Choose number of Closing', (1, 2, 3, 4, 5, 6, 7, 8, 9, 10))

        st.title('Closing Morphological')
        image = load_image()
        useWH = st.button('CONVERT')

        if useWH:
            resized_image = closing(image, selected_box_operation_level)
            st.image(resized_image, caption=f"Image with Closing", use_column_width=True)

    # End Closing

    # Begin Skeletonization

    def skeletonization(original_image, level):
        output_image = morph_thinning_skeletonize(original_image, True, level)

        return output_image

    if selected_box == 'Skeletonization':

        selected_box_operation_level = st.sidebar.selectbox('Choose number of pruning', (5, 10, 15, 20, 25))

        st.title('Skeletonization Morphological')
        image = load_image()
        useWH = st.button('CONVERT')

        if useWH:
            resized_image = skeletonization(image, selected_box_operation_level)
            st.image(resized_image, caption=f"Image with Skeletonization", use_column_width=True)

    # End Skeletonization

    # Begin Border Separation

    def border_separation(original_image, level):
        # Read Image
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(gray_image, cv2.COLOR_BGR2RGB)

        binr = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)[1]

        kernel = np.ones((3, 3), np.uint8)
        erosion_img = cv2.erode(original_image, kernel, iterations=level)
        output_image = cv2.subtract(original_image, erosion_img)
        return output_image

    if selected_box == 'Border Separation':

        selected_box_operation_level = st.sidebar.selectbox('Choose number of the erosion in algorithm',
                                                            (1, 2, 3, 4, 5, 6, 7, 8, 9, 10))

        st.title('Border Separation Morphological')
        image = load_image()
        useWH = st.button('CONVERT')

        if useWH:
            resized_image = border_separation(image, selected_box_operation_level)
            st.image(resized_image, caption=f"Image with Border Separation", use_column_width=True)

    # End Border Separation

    # Begin GRADIENT

    def gradient(original_image, level):
        # Read Image
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(gray_image, cv2.COLOR_BGR2RGB)

        (thresh, blackAndWhiteImage) = cv2.threshold(
            img, 127, 255, cv2.THRESH_BINARY)

        kernel_size = (level, level)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        output_image = cv2.morphologyEx(blackAndWhiteImage, cv2.MORPH_GRADIENT, kernel)

        return output_image

    if selected_box == 'Gradient':

        selected_box_operation_level = st.sidebar.selectbox('Choose one of the level', (1, 2, 3, 4, 5, 6, 7, 8, 9, 10))

        st.title('Gradient Morphological')
        image = load_image()
        useWH = st.button('CONVERT')

        if useWH:
            resized_image = gradient(image, selected_box_operation_level)
            st.image(resized_image, caption=f"Image with Gradient", use_column_width=True)

    # End Gradient

    # Begin TopHat

    def topHat(original_image, level):
        # Read Image
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(gray_image, cv2.COLOR_BGR2RGB)

        kernel_size = (level, level)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        output_image = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

        return output_image

    if selected_box == 'Top Hat':

        selected_box_operation_level = st.sidebar.selectbox('Choose one of the level',
                                                            (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15))

        st.title('Top Hat Morphological')
        image = load_image()
        useWH = st.button('CONVERT')

        if useWH:
            resized_image = topHat(image, selected_box_operation_level)
            st.image(resized_image, caption=f"Image with Top Hat", use_column_width=True)

    # End Top Hat

    # Begin Black Hat

    def blackHat(original_image, level):
        # Read Image
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(gray_image, cv2.COLOR_BGR2RGB)

        kernel_size = (level, level)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        output_image = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

        return output_image

    if selected_box == 'Black Hat':

        selected_box_operation_level = st.sidebar.selectbox('Choose one of the level',
                                                            (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15))

        st.title('Black Hat Morphological')
        image = load_image()
        useWH = st.button('CONVERT')

        if useWH:
            resized_image = blackHat(image, selected_box_operation_level)
            st.image(resized_image, caption=f"Image with Black Hat", use_column_width=True)

    # End Black Hat

    # Begin Region Filling

    def regionFilling(original_image):
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        (thresh, original_image) = cv2.threshold(
            gray_image, 127, 255, cv2.THRESH_BINARY)

        st.markdown('<h3>Image with Region Filling</h3>', unsafe_allow_html=True)
        seeds = np.array([[213, 221], [178, 671], [523, 170], [448, 449], [482, 834]])  # seeds inside of the boundaries
        res = region_filling(original_image, seeds)

        plt.axis('off')
        plt.imshow(res, 'gray')
        region_filling_img = plt.show()
        st.pyplot(region_filling_img)

    if selected_box == 'Region Filling':

        st.title('Region Filling Morphological')
        image = load_image()

        useWH = st.button('CONVERT')

        if useWH:
            regionFilling(image)

    # End Region Filling

    # Begin Extract Components

    def extractComponents(original_image):

        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        (thresh, original_image) = cv2.threshold(
            gray_image, 127, 255, cv2.THRESH_BINARY)

        st.markdown('<h3>Image with Extract Components</h3>', unsafe_allow_html=True)
        res, labels = extract_components(original_image)
        n_components = min(labels, 6)  # show at most 6 components

        ind = 1
        plt.figure(figsize=(20, 10))
        while ind <= n_components:
            plt.subplot(1, n_components, ind)
            plt.title('Label:' + str(ind), fontsize=15)
            plt.axis('off')
            x = np.zeros(original_image.shape, dtype=np.uint8)
            x[res == ind] = 255
            plt.imshow(x, 'gray')
            ind = ind + 1
        extract_components_img = plt.show()
        st.pyplot(extract_components_img)

    if selected_box == 'Extract Components':

        st.title('Extract Components Morphological')
        image = load_image()

        useWH = st.button('CONVERT')

        if useWH:
            extractComponents(image)

    # End Extract Components

    # Begin Convex Hull

    def convexHull(original_image):

        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        (thresh, original_image) = cv2.threshold(
            gray_image, 127, 255, cv2.THRESH_BINARY)

        st.markdown('<h3>Image with Convex Hull</h3>', unsafe_allow_html=True)

        ch = convex_hull(original_image)

        plt.figure(figsize=(24, 24))

        plt.subplot(1, 3, 1)
        plt.title('Original Image', fontsize=22)
        plt.axis("off")
        plt.imshow(original_image, 'gray')

        plt.subplot(1, 3, 2)
        plt.title('Image + Convex Hull', fontsize=22)
        plt.axis("off")
        plt.imshow(ch - original_image, 'gray')

        plt.subplot(1, 3, 3)
        plt.title('Convex Hull', fontsize=22)
        plt.axis("off")
        plt.imshow(ch, 'gray')
        convex_hull_img = plt.show()

        st.pyplot(convex_hull_img)

    if selected_box == 'Convex Hull':

        st.title('Convex Hull Morphological')
        image = load_image()

        useWH = st.button('CONVERT')

        if useWH:
            convexHull(image)

    # End Convex Hull
