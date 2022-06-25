from re import A
import streamlit as st
from PIL import Image
import cv2
import numpy as np
import copy


def app():
    DEMO_IMAGE = 'imgTests/erosion.jpg'

    selected_box = st.sidebar.selectbox('Choose one of the filters', ('None', 'Erosion'))

    def load_image():
        img_file_buffer = st.file_uploader("Upload an image", type=["jpg", "jpeg", 'png'])
        if img_file_buffer is not None:
            image = np.array(Image.open(img_file_buffer))
        else:
            demo_image = DEMO_IMAGE
            image = np.array(Image.open(demo_image))

        st.image(image, caption=f"Original Image", use_column_width=True)
        return image

    def erosion(photo):
        # Read Image
        original_image = photo.copy()
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(gray_image, cv2.COLOR_BGR2RGB)

        (thresh, blackAndWhiteImage) = cv2.threshold(
            img, 127, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)
        output_image = cv2.erode(blackAndWhiteImage, kernel, iterations=2)

        return output_image

    if selected_box == 'None':
        st.title('Morphological')
        ## Add bulletins
        st.subheader("Select from the following morphological", anchor=None)
        st.header("CV", anchor=None)

        st.subheader("Available Filters", anchor=None)

        st.markdown(
            '<ul> <li> Erosion <li> </ul>',
            unsafe_allow_html=True)

    if selected_box == 'Erosion':
        st.title('Erosion Morphological')
        image = load_image()
        useWH = st.button('CONVERT')

        if useWH:
            resized_image = erosion(image)
            st.image(resized_image, caption=f"Image with Erosion", use_column_width=True)
