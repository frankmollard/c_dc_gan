import streamlit as st
import tensorflow as tf
from tensorflow.random import normal
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image 

@st.cache_resource()
def modelloader(name):
    m = tf.keras.models.load_model(name)
    return m

st.sidebar.write('## Configuration')

gen_select = st.sidebar.radio(
    "Choose Generator",
    ["Generator64mw.keras", "Generator64mw_norm_2.keras"]
)

if gen_select in ["Generator128_norm.keras"]:
    imageSize = (128,128,3)
    creatures = {"cat":0, "dog":1, "wild":2}
else:
    imageSize = (64,64,3)
    creatures = {"cat":0, "dog":1, "man":2, "wild":3, "woman":4}

generator = modelloader(gen_select)

st.write("""
# Mix Creatures
""")

st.write('This is the Generator of a Conditional Deep Convolutional GAN')




distribution = st.sidebar.radio(
    "Kind of Distribution for Latent Space",
    ["Binomial", "Normal"],
    captions = ["clearer results", "more variance"]
)

if distribution == "Normal":
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
else:
    m, s = 0, 0
    
option1 = st.sidebar.selectbox(
    'Select Creature',
    #('cat', 'dog', 'man', 'wild', 'woman')
    creatures.keys() 
)

option2 = st.sidebar.selectbox(
    'Select 2nd Creature',
    ('None', 'cat', 'dog', 'man', 'wild', 'woman')
)

if option2 == 'None':
    option2 = option1

#@st.cache()
def vizGenImg(shp, g, sd, me, dist, c1: int, c2: int, ttl: str):
    fig, ax = plt.subplots(1,5, figsize=(15*2,3*2))
    for j in range(5):
        if dist == "Binomial":
            xx = np.random.randint(2, size=(1, 100))
            xx = xx.reshape(-1,100)
        else:
            xx = normal(shape=(1, 100), mean=me, stddev=sd)
        xc = tf.stack([[creatures[c1]],[creatures[c2]]], axis=1)
        img = g.predict([xx, xc], verbose=0)
        img = Image.fromarray(((img+1)/2*255).astype('uint8').reshape(shp))
        ax[j].axis('off')
        ax[j].imshow(img)
        ax[j].set_title(ttl, fontsize=25)
    fig.tight_layout(pad=0)
    return fig, ax, plt 

if st.sidebar.button("Run", type="primary"):
    for k in range(3):
        image, _, _=vizGenImg(imageSize, generator, s, m, distribution, option1, option2, ttl=option1 if option1 ==option2 else option1 + " " + option2)
        st.pyplot(fig=image)
