from re import A
import streamlit as st
from PIL import Image
import cv2
import numpy as np
import copy

from apps.morphological_skeletonization import morph_thinning_skeletonize


def app():
    selected_box = st.sidebar.selectbox('Choose one of the operations',
                                        ('None', 'Erosion', 'Dilation', 'Opening', 'Closing', 'Skeletonization'))

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

        st.subheader("Available Filters", anchor=None)

        st.markdown(
            '<ul> <li> Erosion <li> Dilation <li> Opening <li> Closing <li> Skeletonization </ul>',
            unsafe_allow_html=True)

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

        selected_box_operation_level = st.sidebar.selectbox('Choose one of the level', (1, 2, 3, 4, 5, 6, 7, 8, 9, 10))

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

        selected_box_operation_level = st.sidebar.selectbox('Choose one of the level', (1, 2, 3, 4, 5, 6, 7, 8, 9, 10))

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

        selected_box_operation_level = st.sidebar.selectbox('Choose one of the level', (1, 2, 3, 4, 5, 6, 7, 8, 9, 10))

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

        selected_box_operation_level = st.sidebar.selectbox('Choose one of the level', (1, 2, 3, 4, 5, 6, 7, 8, 9, 10))

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

        selected_box_operation_level = st.sidebar.selectbox('Choose one of the level', (5, 10, 15, 20, 25))

        st.title('Skeletonization Morphological')
        image = load_image()
        useWH = st.button('CONVERT')

        if useWH:
            resized_image = skeletonization(image, selected_box_operation_level)
            st.image(resized_image, caption=f"Image with Skeletonization", use_column_width=True)

    # End Skeletonization
