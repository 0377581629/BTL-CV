import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import cv2
from multiapp import MultiApp
from apps import home, morphological


app = MultiApp()


# option = st.selectbox(
#     'Select from the options',
#     ('Home', 'Filters', 'Doc scanner','add text'), key = 1)


# if(option=='Filters'):
#     opt = st.selectbox(
#     'Select from the options',
#     ('sepia', 'Filter1', 'filter2','filter3'), key = 2)

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Morphological", morphological.app)


# The main app
app.run()