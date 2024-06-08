import streamlit as st
from glob import glob
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import pandas as pd
import main

plt.rcParams['image.cmap'] = 'gray'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if 'clicked' not in st.session_state:
    st.session_state.clicked = False

def click_button():
    st.session_state.clicked = True
    
def click_stop():
    st.session_state.clicked = False
    
st.title("Create and test new metric")

data_encoder = ['InceptionV3', 'CLIP', 'DINOv2']
encoder = st.selectbox('Metric encoder', data_encoder)

data_formula = ('FD', 'MMD')
formula = st.selectbox('Metric formula based on', data_formula)

data_class_type = ('No', 'Yes')
class_type = st.selectbox('Metric class averaging', data_class_type)

sample_size_params = (range(10, 51, 10))
sample_size = st.selectbox('Sample size', sample_size_params)

data_effect = ('Noise', 'Sheltering', 'Exchange', 'No')
effect = st.selectbox('Effect', data_effect)

data_power = ('Low', 'Middle', 'High', 'Extra', 'No')
power = st.selectbox('Power', data_power)

real_image_path = st.text_input("Real image path", "real_images")
generated_image_path = st.text_input("Generated image path", "generated_images")

st.button('OK', on_click=click_button)

if st.session_state.clicked:
    
    p = '/home/samsmu/Code/VKR/cmmd-pytorch/reference_images/'
    s = '/home/samsmu/Code/VKR/cmmd-pytorch/generated_images/'
    
    #ans = main.run(encoder, class_type, formula, sample_size, effect, power, real_image_path, generated_image_path)
    ans = main.run(encoder, class_type, formula, sample_size, effect, power, p, s)
    
    st.write(f"Value of metric is {ans}")
