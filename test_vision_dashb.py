import pickle

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

import cv2 as cv

import matplotlib.pyplot as plt

import argparse

import streamlit as st

import argparse


parser = argparse.ArgumentParser(
    description='training')

parser.add_argument(
    '-dn',
    '--dataset_name',
    default='sports')#eurosat,satelimgslocs,sports

parser.add_argument(
    '-res','--resolution',
    type=int,
    default=64)



inputs=parser.parse_args()

dataset_name=inputs.dataset_name
res=inputs.resolution


# dataset_name = st.text_input("dataset_name", "sports")
# res = int(st.text_input("resolution", "64"))



# File uploader widget
uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])


@st.cache_data
def prepare(dn):

    new_model = tf.keras.models.load_model(f'vision/models/my_model_{dataset_name}{res}.keras')

    # Show the model architecture
    new_model.summary()

    with open(f"vision/output_classification_preparation/{dataset_name}_unique_labels.pkl", "rb") as f:
        unique_labels = pickle.load(f)
    return new_model,unique_labels

new_model,unique_labels=prepare('sports')



if uploaded_file is not None:
    # Display file name
    st.write("Filename:", uploaded_file.name)

    # Optionally, read contents
    file_content = uploaded_file.read()
    st.write("File size (bytes):", len(file_content))

    image_rgb = tf.image.decode_image(file_content)

    image_rgb = tf.image.resize(image_rgb, [res, res], method='nearest')

    st.image(image_rgb.numpy(), caption="Uploaded Image", width=350)

    image_rgb = tf.cast(image_rgb, tf.float32)

    n_img=preprocess_input(image_rgb)

    n_img=tf.expand_dims(n_img,0)


# image = cv.imread(path)
#
# # converting BGR to RGB
# image_rgb1 = cv.cvtColor(image, cv.COLOR_BGR2RGB)
# image_rgb1=tf.convert_to_tensor(image_rgb1, dtype=tf.float32)


    #import numpy as np
    #tf.nn.softmax(new_model.predict(n_img))

    vs=tf.nn.softmax(new_model.predict(n_img))

    mv=tf.math.reduce_max(vs,axis=-1)

    mv = tf.squeeze(mv)

    ind=tf.math.argmax(vs,axis=-1)

    ind=tf.squeeze(ind)


    st.write(f"Our prediction is {unique_labels[ind]} and also {100*mv:.2f}% sure.")

    unique_labels


