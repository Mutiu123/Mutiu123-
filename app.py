#import library
import numpy as np
import pickle as pkl
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPool2D

from sklearn.neighbors import NearestNeighbors
import os
from numpy.linalg import norm
import streamlit as st 

st.header('Fashion Recommendation System')

image_features = pkl.load(open('model/image_features.pkl','rb'))
datanames = pkl.load(open('model/datanames.pkl','rb'))

def extract_features_from_images(image_path, model):
    img = image.load_img(image_path, target_size=(224,224)) #load image with terget size
    image_array = image.img_to_array(img)                   #converting it to the array
    img_expand_dim = np.expand_dims(image_array, axis=0)    # reshape the image into 4 dimmension using expand_dims
    img_prepocess = preprocess_input(img_expand_dim)        #conver the image from RGB configuration to BRG configuration before feeding it into the model
    results = model.predict(img_prepocess).flatten()         #Predict extracted feature and flatten the results/array
    norm_results = results/norm(results)                      #Normalise the result
    return norm_results   


model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
model.trainable = False

model = tf.keras.Sequential([
    model,
    GlobalMaxPool2D()
]) 

neighbors = NearestNeighbors(n_neighbors=7, algorithm='brute', metric='euclidean') 
neighbors.fit(image_features) 


upload_file = st.file_uploader("Please Select Product of Your Choice")
if upload_file is not None:
    with open(os.path.join('Selected_products', upload_file.name), 'wb') as f:
        f.write(upload_file.getbuffer())
    st.subheader("Selected Product")
    st.image(upload_file)
    inpu_image_features = extract_features_from_images(upload_file, model) 
    distance, indices = neighbors.kneighbors([inpu_image_features])
    st.subheader('Recommended Items')
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1: 
        st.image(datanames[indices[0][1]])
    with col2: 
        st.image(datanames[indices[0][2]])
    with col3: 
        st.image(datanames[indices[0][3]])
    with col4: 
        st.image(datanames[indices[0][4]])
    with col5: 
        st.image(datanames[indices[0][5]])
    with col6: 
        st.image(datanames[indices[0][6]])
    