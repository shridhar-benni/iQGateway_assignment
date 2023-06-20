import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

# CSS styling for the drag and drop box
css = """
    .file-drop-area {
        min-height: 200px;
        line-height: 200px;
        text-align: center;
        background-color: #f5f5f5;
        border: 2px dashed #cccccc;
        border-radius: 10px;
        padding: 5px;
    }
"""

# Apply the CSS styling
st.write(f'<style>{css}</style>', unsafe_allow_html=True)

# Create the drag and drop box
uploaded_file = st.file_uploader("Drag and drop an image here", type=["png", "jpg", "jpeg"])

# Check if an image file is uploaded
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    # Resize the image to 150x150 pixels
    resized_image = image.resize((150, 150))
    # Convert the resized image to a NumPy array
    test_img = np.array(resized_image)
    test_ia = test_img.reshape((1, 150, 150, 3))
    model = load_model('best_model.h5')
    if model.predict(test_ia)[0][0] > 0.5:
        st.write("Infected")
    else:
        st.write("Uninfected")
    with tf.GradientTape() as tape:
        last_conv_layer = model.get_layer('conv2d_2')
        iterate = tf.keras.models.Model([model.inputs], [model.output, last_conv_layer.output])
        model_out, last_conv_layer = iterate(test_ia)
        class_out = model_out[:, np.argmax(model_out[0])]
        grads = tape.gradient(class_out, last_conv_layer)
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, last_conv_layer), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = heatmap.reshape((34, 34))
    heatmap_plt = cv2.resize(heatmap , (150,150))
    # Apply a color mapping scheme (e.g., 'hot') to convert the heat map to RGB
    heatmap_rgb = plt.get_cmap('jet')(heatmap_plt)[:, :, :3]  # Keep only RGB channels #viridis
    # Resize the overlay image to match the background image size
    overlay_image = cv2.resize(heatmap_rgb*255, (test_img.shape[1], test_img.shape[0]))
    overlay_image = overlay_image.astype(test_img.dtype)
    # Overlay the images by adding the pixel values
    blended_image = cv2.addWeighted(test_img, 0.5, overlay_image, 0.5, 0)
    st.image(blended_image, use_column_width=True, caption="Uploaded and Resized Image")
