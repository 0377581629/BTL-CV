from matplotlib import image
import streamlit as st
from PIL import Image
import cv2
import numpy as np
from streamlit_cropper import st_cropper
import copy

from apps.arithmetic_operations.rotation import *
from apps.arithmetic_operations.convert_to_binary import *
from apps.arithmetic_operations.filter import *


def app():
    selected_box = st.sidebar.selectbox('Choose one of the operations',
                                        ('None', 'Addition', 'Subtraction', 'Cropping', 'Resizing', 'Bitwise', 'Affine',
                                         'Rotating', 'Flipping', 'Filters',))

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
            '<ul> <li> Addition <li> Subtraction <li> Cropping <li> Resizing <li> Bitwise <li> Affine <li> Rotating <li> Flipping <li> Black Hat <li> Region Filling <li> Extract Components <li> Convex Hull </ul>',
            unsafe_allow_html=True)

    # Begin Addition

    if selected_box == 'Addition':

        st.title('Addition Arithmetic')
        image1 = load_image1()
        image2 = load_image2()
        useWH = st.button('RESULT')

        if useWH:
            output_image = cv2.add(image1, image2)
            st.image(output_image, caption=f"Image with Addition", use_column_width=True)

    # Begin Subtraction

    if selected_box == 'Subtraction':

        st.title('Subtraction Arithmetic')
        image1 = load_image1()
        image2 = load_image2()
        useWH = st.button('RESULT')

        if useWH:
            output_image = cv2.subtract(image1, image2)
            st.image(output_image, caption=f"Image with Subtraction", use_column_width=True)

    # Begin Cropping

    if selected_box == 'Cropping':
        st.title("Cropping Image")
        img_file = st.file_uploader(label='Upload a file', type=['png', 'jpg', 'jpeg'])
        realtime_update = st.sidebar.checkbox(label="Update in Real Time", value=True)
        box_color = st.color_picker(label="Box Color", value='#0000FF')
        aspect_choice = st.sidebar.radio(label="Aspect Ratio", options=["1:1", "16:9", "4:3", "2:3", "Free"])
        aspect_dict = {"1:1": (1, 1),
                       "16:9": (16, 9),
                       "4:3": (4, 3),
                       "2:3": (2, 3),
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
            _ = cropped_img.thumbnail((150, 150))
            st.image(cropped_img)

    # End Cropping

    # Begin Resizing

    def process_image(image, points):
        resized_img = cv2.resize(image, points, interpolation=cv2.INTER_LINEAR)
        return resized_img

    @st.cache
    def process_scaled_image(image, scaling_factor):
        resized_img = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_LINEAR)
        return resized_img

    if selected_box == 'Resizing':
        st.title('Resizing Image')
        img = load_image1()
        useWH = st.checkbox('Resize using a Custom Height and Width')
        useScaling = st.checkbox('Resize using a Scaling Factor')

        if useWH:
            st.subheader('Input a new Width and Height')
            width = int(st.number_input('Input a new a Width', value=720))
            height = int(st.number_input('Input a new a Height', value=720))
            points = (width, height)
            resized_image = process_image(img, points)

            st.image(
                resized_image, caption=f"Resized image", use_column_width=False)

        if useScaling:
            st.subheader('Drag the Slider to change the Image Size')
            scaling_factor = st.slider('Reszie the image using scaling factor', min_value=0.1, max_value=5.0,
                                       value=1.0, step=0.5)
            resized1_image = process_scaled_image(img, scaling_factor)
            st.image(
                resized1_image, caption=f"Resized image using Scaling factor", use_column_width=False)

    # End Resizing

    # def load_image_rotation():
    #     file_uploaded = st.file_uploader(label='Upload a file', type=['png', 'jpg','jpeg'])
    #     if file_uploaded is not None:
    #         image = Image.open(file_uploaded)
    #         image_cv2 = np.array(image)
    #         image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2BGRA)

    # Begin Affine

    if selected_box == 'Affine':
        st.title('Affine')
        file_uploaded = st.file_uploader(label='Upload a file', type=['png', 'jpg', 'jpeg'])
        if file_uploaded is not None:
            image = np.array(Image.open(file_uploaded))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
            st.image(image, caption=f"Original Image", use_column_width=True)
        else:
            return None
        st.markdown("Image after **affine transformation** :wave: ...")
        st.image(warpaffine(image))

    # End Affine

    # Begin Rotating

    if selected_box == 'Rotating':
        st.title('Rotating Image')
        file_uploaded = st.file_uploader(label='Upload a file', type=['png', 'jpg', 'jpeg'])
        if file_uploaded is not None:
            image = np.array(Image.open(file_uploaded))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
            st.image(image, caption=f"Original Image", use_column_width=True)
        else:
            return None
        ang = st.slider("select the angle of rotation...", min_value=0, max_value=360)
        st.write(f"Slider value: {ang} degrees")
        st.markdown("Image after **rotation**...")
        st.image(rotating(image, ang))

    # End Rotating

    # Begin Flipping

    if selected_box == 'Flipping':
        st.title('Flipping Image')
        file_uploaded = st.file_uploader(label='Upload a file', type=['png', 'jpg', 'jpeg'])
        if file_uploaded is not None:
            image = np.array(Image.open(file_uploaded))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
            st.image(image, caption=f"Original Image", use_column_width=True)
        else:
            return None
        flip_out = st.selectbox("Select an option: ", ('Select', 0, 1, -1))
        st.write("You selected: ", flip_out)
        if flip_out == 'Select':
            pass
        else:
            st.header("Flipped image...")
            st.image(flipping(image, flip_out))

    # End Flipping

    # Begin Bitwise

    if selected_box == 'Bitwise':
        option = st.sidebar.selectbox('Choose one of the bitwise operations',
                                      ('AND', 'OR', 'XOR', 'NOT',))
        image1 = load_image1()
        image2 = load_image2()

        if option == 'AND':
            st.title('AND operations')
            useWH = st.button('CONVERT')
            if useWH:
                final_AND = cv2.bitwise_and(image1, image2, mask=None)
                st.image(final_AND, caption=f"Converted image", use_column_width=True)

        if option == 'OR':
            st.title('OR operations')
            useWH = st.button('CONVERT')
            if useWH:
                final_OR = cv2.bitwise_or(image1, image2, mask=None)
                st.image(final_OR, caption=f"Converted image", use_column_width=True)

        if option == 'XOR':
            st.title('XOR operations')
            useWH = st.button('CONVERT')
            if useWH:
                final_XOR = cv2.bitwise_xor(image1, image2, mask=None)
                st.image(final_XOR, caption=f"Converted image", use_column_width=True)

        if option == 'NOT':
            st.title('NOT operations')
            useWH = st.button('CONVERT')
            if useWH:
                final_NOT = cv2.bitwise_not(image1, mask=None)
                st.image(final_NOT, caption=f"Converted image", use_column_width=True)

    # End Bitwise

    if selected_box == 'Filters':
        option = st.sidebar.selectbox('Choose one of filters',
                                      ('None', 'Bright', 'Detail Enchance', 'Invert', 'Summer', 'Winter', 'Daylight',
                                       'High Contrast', 'Sepia', 'Splash', 'Emboss', '60s TV', 'Dual Tone', 'Cartoon',
                                       'Pencil Drawing', 'Comic'))

        if option == 'None':
            st.title('Image Filters')
            ## Add bulletins
            st.subheader("Select from the following filters", anchor=None)
            st.header(" &#128072", anchor=None)

            st.subheader("Available Filters", anchor=None)

            st.markdown(
                '<ul> <li> Bright <li>  Detail Enchance  <li> Invert<li> Summer   <li> Winter  <li> Daylight <li> High Contrast<li> Sepia  <li> Splash<li> Emboss   <li> 60s TV  <li> Dual Tone <li> Cartoon <li>Pencil Drawing <li>Comic </ul>',
                unsafe_allow_html=True)

        if option == 'Bright':
            st.title('Bright Filter')
            image = load_image1()
            useWH = st.button('CONVERT')

            if useWH:
                resized_image = img2bright(image)
                st.image(resized_image, caption=f"Image with Bright Filter", use_column_width=True)

        if option == 'Detail Enchance':
            st.title('Detail Enchancement')
            image = load_image1()
            useWH = st.button('CONVERT')
            if useWH:
                dst, dst2 = img2enh(image)
                st.image(dst, caption=f"Detail Enhance", use_column_width=True)
                st.image(dst2, caption=f"Kernal Sharpening", use_column_width=True)

        if option == 'Invert':
            st.title('Invert Image')
            image = load_image1()
            useWH = st.button('CONVERT')
            if useWH:
                res = img2inv(image)
                st.image(res, caption=f"Inverted Image", use_column_width=True)

        if option == 'Summer':
            st.title('Summer Filter')
            image = load_image1()

            useWH = st.button('CONVERT')
            if useWH:
                res = img2sum(image)
                st.image(res, caption=f"Image with Summer Filter", use_column_width=True)

        if option == 'Winter':
            st.title('Winter Filter')
            image = load_image1()

            useWH = st.button('CONVERT')
            if useWH:
                res = img2win(image)
                st.image(res, caption=f"Image with Winter Filter", use_column_width=True)

        if option == 'Daylight':
            st.title('Daylight Filter')
            image = load_image1()

            useWH = st.button('CONVERT')
            if useWH:
                res = img2day(image)
                st.image(res, caption=f"Image with Daylight Filter", use_column_width=True)

        if option == 'High Contrast':
            st.title('High Contrast Filter')
            image = load_image1()

            useWH = st.button('CONVERT')
            if useWH:
                res = img2cont(image)
                st.image(res, caption=f"Image with High Contrast", use_column_width=True)

        if option == 'Sepia':
            st.title('Sepia Filter')
            image = load_image1()

            useWH = st.button('CONVERT')
            if useWH:
                resized_image = img2sepia(image)
                st.image(resized_image, caption=f"Image with Sepia Filter", use_column_width=True)

        if option == 'Splash':
            SP_DEMO_IMAGE = 'imgs/ball.jpg'
            SP_IMAGE = 'imgs/Splash.jpg'
            st.title('Splash Filter')

            img_file_buffer = st.file_uploader("Upload an image", type=["jpg", "jpeg", 'png'])
            if img_file_buffer is not None:
                image = np.array(Image.open(img_file_buffer))
                st.image(image, caption=f"Original Image", use_column_width=True)
            else:
                demo_image = SP_DEMO_IMAGE
                image = np.array(Image.open(demo_image))

            useWH = st.button('CONVERT')
            if useWH:
                resized_image = img2splash(image)
                splash_image = SP_IMAGE
                resized_image = np.array(Image.open(splash_image))
                st.image(resized_image, caption=f"Image with Splash Filter", use_column_width=True)

        if option == 'Emboss':
            st.title('Emboss Filter')
            image = load_image1()

            useWH = st.button('CONVERT')
            if useWH:
                resized_image = img2emb(image)
                st.image(resized_image, caption=f"Image with Emboss Filter", use_column_width=True)

        if option == '60s TV':
            st.title('60s TV Filter')
            image = load_image1()
            res = tv_60(image)

        if option == 'Dual Tone':
            st.title('Dual Tone Filter')
            image = load_image1()

            im1 = copy.deepcopy(image)
            im2 = copy.deepcopy(image)
            im3 = copy.deepcopy(image)

            useWH = st.button('CONVERT')
            if useWH:
                r1 = img2tone(im1, 0)
                st.image(r1, caption=f"Dual Tone with Red Channel", use_column_width=True)
                r2 = img2tone(im2, 1)
                st.image(r2, caption=f"Dual Tone with Green Channel", use_column_width=True)
                r3 = img2tone(im3, 2)
                st.image(r3, caption=f"Dual Tone with Blue Channel", use_column_width=True)

        if option == 'Cartoon':
            st.title('Cartoon Filter')
            image = load_image1()

            useWH = st.button('CONVERT')
            if useWH:
                resized_image = img2cartoon(image)
                st.image(resized_image, caption=f"Image with Cartoon Filter", use_column_width=True)

        if option == 'Pencil Drawing':
            st.title('Pencil Drawing Filter')
            image = load_image1()

            useWH = st.button('CONVERT')
            if useWH:
                resized_image = img2pen(image)
                st.image(resized_image, caption=f"Image with Pencil Drawing Filter", use_column_width=True)

        if option == 'Comic':
            st.title('Comic Filter Using K-Means')
            image = load_image1()
            res = comic(image)
