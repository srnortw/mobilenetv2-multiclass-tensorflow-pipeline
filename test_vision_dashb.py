import pickle#test from colab

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

import cv2 as cv

import matplotlib.pyplot as plt

import argparse

import streamlit as st

import argparse

import os

# parser = argparse.ArgumentParser(
#     description='training')
#
# parser.add_argument(
#     '-dn',
#     '--dataset_name',
#     default='sports')#eurosat,satelimgslocs,sports
#
# parser.add_argument(
#     '-res','--resolution',
#     type=int,
#     default=224)
#
#
#
# inputs=parser.parse_args()
#
# dataset_name=inputs.dataset_name
# res=inputs.resolution


# dataset_name = st.text_input("dataset_name", "sports")
# res = int(st.text_input("resolution", "64"))

# File uploader widget
uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])


@st.cache_data
def prepare(model_name):

    new_model = tf.keras.models.load_model(f'vision/models/{model_name}.keras')

    # Show the model architecture
    new_model.summary()

    with open(f"vision/unique_labels_folder/{model_name.split("_")[-2]}_unique_labels.pkl", "rb") as f:
        unique_labels = pickle.load(f)

    return new_model,unique_labels

model_files = [f[:-6] for f in os.listdir('vision/models') if f.endswith(".keras")]

model_name= st.sidebar.radio("Pick A Model",model_files)

new_model,unique_labels=prepare(model_name)


@tf.function
def enter(image_rgb_uint):

    image_rgb = tf.cast(image_rgb_uint, tf.float32)

    n_img = preprocess_input(image_rgb)

    n_img = tf.expand_dims(n_img, 0)

    vs = tf.nn.softmax(new_model(n_img, training=False))  # Replace .predict
    mv = tf.reduce_max(vs, axis=-1)
    ind = tf.argmax(vs, axis=-1)

    return tf.squeeze(mv), tf.squeeze(ind)


if uploaded_file is not None:
    # Display file name

    file_content = uploaded_file.read()


    image_rgb_uint = tf.image.decode_image(file_content)

    input_shape=new_model.input_shape[1:-1]

    image_rgb_uint = tf.image.resize(image_rgb_uint, input_shape, method='nearest')

    right,left=st.columns(2)

    with right:

        # st.write("Filename:", uploaded_file.name)
        # # Optionally, read contents
        # st.write("File size (bytes):", len(file_content))

        st.write(f"Resizing image to {input_shape}")

        st.image(image_rgb_uint.numpy(), caption="Uploaded Image", width=350)

        mv, ind = enter(image_rgb_uint)

        st.write(f"Our prediction is {unique_labels[ind]} with {100*mv:.2f}% confidence.")

    with left:

        unique_labels




# image = cv.imread(path)
#
# # converting BGR to RGB
# image_rgb1 = cv.cvtColor(image, cv.COLOR_BGR2RGB)
# image_rgb1=tf.convert_to_tensor(image_rgb1, dtype=tf.float32)


    #import numpy as np
    #tf.nn.softmax(new_model.predict(n_img))


