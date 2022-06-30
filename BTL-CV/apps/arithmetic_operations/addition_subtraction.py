import streamlit as st
from PIL import Image
import cv2
import numpy as np

def app():
    selected_box = st.sidebar.selectbox('Choose one of the operations',
                                        ('None', 'Addition', 'Subtraction', 'Opening', 'Closing', 'Skeletonization',
                                         'Border Seperation', 'Gradient', 'Top Hat', 'Black Hat', 'Region Filling',
                                         'Extract Components','Convex Hull'))

    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Begin upload img

    def load_image1():
        img_file_buffer1 = st.file_uploader("Upload an image", type=["jpg", "jpeg", 'png'])
        if img_file_buffer1 is not None:
            image = np.array(Image.open(img_file_buffer1))
            st.image(image, caption=f"Original Image", use_column_width=True)
            return image
        else:
            return None

    def load_image2():
        img_file_buffer2 = st.file_uploader("Upload another image", type=["jpg", "jpeg", 'png'])
        if img_file_buffer2 is not None:
            image0 = np.array(Image.open(img_file_buffer2))
            st.image(image0, caption=f"Original Image", use_column_width=True)
            return image0
        else:
            return None

    # End upload img

    if selected_box == 'None':
        st.title('Arithmetic')
        ## Add bulletins
        st.subheader("Select from the following arithmetic", anchor=None)
        st.header("CV", anchor=None)

        st.subheader("Available arithmetic Operations", anchor=None)

        st.markdown(
            '<ul> <li> Addition <li> Dilation <li> Opening <li> Closing <li> Skeletonization <li> Border Seperation <li> Gradient <li> Top Hat <li> Black Hat <li> Region Filling <li> Extract Components <li> Convex Hull </ul>',
            unsafe_allow_html=True)

    if selected_box == 'Addition':

        st.title('Addition Arithmetic')
        image1 = load_image1()
        image2 = load_image2()
        useWH = st.button('CONVERT')

        if useWH:
            output_image = cv2.add(image1, image2)
            st.image(output_image, caption=f"Image with Addition", use_column_width=True)

    if selected_box == 'Subtraction':

        st.title('Subtraction Arithmetic')
        image1 = load_image1()
        image2 = load_image2()
        useWH = st.button('CONVERT')

        if useWH:
            output_image = cv2.subtract(image1, image2)
            st.image(output_image, caption=f"Image with Subtraction", use_column_width=True)
