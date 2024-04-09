#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing import image

# Define the Streamlit app
def app():


    st.subheader('Testing the Performance of the CNN Classification Model')
    text = """We test our trained model by presenting it with a classification task."""
    st.write(text)

    # Create a file uploader widget
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        present_image(uploaded_file)

def decode_one_hot(result):
  """Decodes a one-hot encoded vector to its class label.

  Args:
    result: A numpy array representing a one-hot encoded vector.

  Returns:
    A string representing the class label. ("rock", "paper", or "scissors")
  """
  index = np.argmax(result)
  classes = {0: "paper", 1: "rock", 2: "scissors"}
  return classes[index]

def present_image(imagefile):
    classifier = st.session_state.classifier
    training_set = st.session_state.training_set
    st.image(imagefile, caption='Uploaded test image')
    test_image = image.load_img(imagefile, target_size=(200, 300))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = classifier.predict(test_image)
    
    # check the one hot encoded result
    # st.write(f'result = {result}')
    # display the indices and classes
    #training_set.class_indices
    prediction = decode_one_hot(result)

    st.subheader('CNN says the image is ' + prediction)
 



#run the app
if __name__ == "__main__":
    app()
