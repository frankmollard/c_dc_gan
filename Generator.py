import streamlit as st
import tensorflow as tf
from tensorflow.random import normal
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image 

generator = tf.keras.models.load_model(r"C:\Users\Admin\Downloads\Generator64mw.keras")

st.write("""
# Mix Creatures
""")

st.write('This is the Generator of a conditional Deep Convolutional GAN')

creatures = {"cat":0, "dog":1, "man":2, "wild":3, "woman":4}

m = st.sidebar.select_slider(
    'Select mean of latent space',
    options=[-0.1, 0, 0.1],
    value = 0
)

s = st.sidebar.select_slider(
    'Select standard deviation of latent space',
    options=np.linspace(2,0.1,20), 
    value = 1
)

option1 = st.sidebar.selectbox(
    'Select Monster',
    #('cat', 'dog', 'man', 'wild', 'woman')
    creatures.keys() 
)

option2 = st.sidebar.selectbox(
    'Select 2nd Monster',
    ('None', 'cat', 'dog', 'man', 'wild', 'woman')
)

#st.write('You selected:', option1)
#st.write('You selected:', option2)

if option2 == 'None':
    option2 = option1


def vizGenImg(g, sd, me, c1: int, c2: int, ttl: str):
    fig, ax = plt.subplots(1,5, figsize=(15*2,3*2))
    for j in range(5):
        #xx = np.random.randint(2, size=(1, 100))
        xx = normal(shape=(1, 100), mean=me, stddev=sd)
        xc = tf.stack([[creatures[c1]],[creatures[c2]]], axis=1)
        img = g.predict([xx, xc], verbose=0)#.reshape(-1,100)
        img = Image.fromarray(((img+1)/2*255).astype('uint8').reshape((64,64,3)))
        ax[j].axis('off')
        ax[j].imshow(img)
        ax[j].set_title(ttl, fontsize=25)
    fig.tight_layout(pad=0)
    return fig, ax, plt 

if st.sidebar.button("Run", type="primary"):
    for k in range(5):
        image, _, _=vizGenImg(generator, s, m, option1, option2, ttl=option1 if option1 ==option2 else option1 + " " + option2)
        st.pyplot(fig=image)