import streamlit as st
from PIL import Image
def app():
    image = Image.open('imgs/lake.jpeg')
    st.image(image, caption='Welcome to our webapp!', use_column_width=True)
    st.subheader('Computer Vision')
    st.subheader("Topic 4", anchor=None)
    st.subheader("More neighbor operator", anchor=None)
    st.subheader("Team Members", anchor=None)
    st.markdown('Đặng Quang Huy   &emsp;&emsp; &emsp; &emsp;20183766  <br> Nguyễn Văn Khang &emsp; &emsp;20183722  <br> Vũ Tùng Dương &nbsp;&nbsp;&emsp; &emsp;20183728  <br> Nguyễn Hoàng Long &nbsp;&nbsp;&emsp; &emsp;20183790  <br> Lương Hoàng Lâm &nbsp;&nbsp;&emsp; &emsp;20183780', unsafe_allow_html=True)
    st.subheader("Submitted to", anchor=None)
    st.markdown("PGS.TS Nguyễn Thị Hoàng Lan")
