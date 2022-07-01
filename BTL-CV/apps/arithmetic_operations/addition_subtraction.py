from matplotlib import image
import streamlit as st
from PIL import Image
import cv2
import numpy as np
from streamlit_cropper import st_cropper

from rotation import *
from filter import *

def app():
    selected_box = st.sidebar.selectbox('Choose one of the operations',
                                        ('None', 'Addition', 'Subtraction', 'Cropping', 'Resizing', 'Bitwise', 'Affine', 'Rotating', 'Flipping',))

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
            '<ul> <li> Addition <li> Subtraction <li> Opening <li> Closing <li> Skeletonization <li> Border Seperation <li> Gradient <li> Top Hat <li> Black Hat <li> Region Filling <li> Extract Components <li> Convex Hull </ul>',
            unsafe_allow_html=True)

    #Begin Addition

    if selected_box == 'Addition':

        st.title('Addition Arithmetic')
        image1 = load_image1()
        image2 = load_image2()
        useWH = st.button('CONVERT')

        if useWH:
            output_image = cv2.add(image1, image2)
            st.image(output_image, caption=f"Image with Addition", use_column_width=True)

    #Begin Subtraction

    if selected_box == 'Subtraction':

        st.title('Subtraction Arithmetic')
        image1 = load_image1()
        image2 = load_image2()
        useWH = st.button('CONVERT')

        if useWH:
            output_image = cv2.subtract(image1, image2)
            st.image(output_image, caption=f"Image with Subtraction", use_column_width=True)

    #Begin Cropping

    if selected_box == 'Cropping':
        st.header("Cropper Demo")
        img_file = st.file_uploader(label='Upload a file', type=['png', 'jpg','jpeg'])
        realtime_update = st.sidebar.checkbox(label="Update in Real Time", value=True)
        box_color = st.color_picker(label="Box Color", value='#0000FF')
        aspect_choice = st.sidebar.radio(label="Aspect Ratio", options=["1:1", "16:9", "4:3", "2:3", "Free"])
        aspect_dict = {"1:1": (1,1),
                      "16:9": (16,9),
                      "4:3": (4,3),
                      "2:3": (2,3),
                      "Free": None}
        aspect_ratio = aspect_dict[aspect_choice]

        if img_file:
            img = Image.open(img_file)
            if not realtime_update:
                st.write("Double click to save crop")
            # Get a cropped image from the frontend
            cropped_img = st_cropper(img, realtime_update=realtime_update, box_color=box_color,
                                        aspect_ratio=aspect_ratio)
        
          # Manipulate cropped image at will
            st.write("Preview")
            _ = cropped_img.thumbnail((150,150))
            st.image(cropped_img)
    
    #End Cropping

    #Begin Resizing

    def process_image(image,points):
    
        resized_img = cv2.resize(image , points , interpolation = cv2.INTER_LINEAR)
    
        return resized_img

    @st.cache
    def process_scaled_image(image,scaling_factor):
    
        resized_img = cv2.resize(image , None,fx= scaling_factor, fy= scaling_factor, interpolation = cv2.INTER_LINEAR)
    
        return resized_img

    if selected_box == 'Resizing':
        st.title('Image Resizing with OpenCV')
        img = load_image1()
        useWH = st.checkbox('Resize using a Custom Height and Width')
        useScaling = st.checkbox('Resize using a Scaling Factor')

        if useWH:
            st.subheader('Input a new Width and Height')

            width = int(st.number_input('Input a new a Width',value = 720))
            height = int(st.number_input('Input a new a Height',value = 720))
            points = (width, height)
            resized_image = process_image(img , points)

            st.image(
            resized_image, caption=f"Resized image", use_column_width=False)
    

        if useScaling:
            st.subheader('Drag the Slider to change the Image Size')
    
            scaling_factor = st.slider('Reszie the image using scaling factor',min_value = 0.1 , max_value = 5.0 ,
                               value = 1.0, step = 0.5)
    
            resized1_image = process_scaled_image(img,scaling_factor)
    
            st.image(
            resized1_image, caption=f"Resized image using Scaling factor", use_column_width=False)
    
    #End Resizing

    #Begin Affine

    if selected_box == 'Affine':
        st.title('Affine')
        image = load_image1(image = cv2.cvtColor(image, cv2.COLOR_BG2BGRA))
        st.markdown("Image after **affine transformation** :wave: ...")
        st.image(warpaffine((image)))
