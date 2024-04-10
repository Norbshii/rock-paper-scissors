#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import time

# Define the Streamlit app
def app():
    if "X" not in st.session_state: 
        st.session_state.X = []
    
    if "y" not in st.session_state: 
        st.session_state.y = []

    if "model" not in st.session_state:
        st.session_state.model = []

    if "X_train" not in st.session_state:
        st.session_state.X_train = []

    if "X_test" not in st.session_state:
            st.session_state.X_test = []

    if "y_train" not in st.session_state:
            st.session_state.y_train = []

    if "y_test" not in st.session_state:
            st.session_state.y_test = []

    if "X_test_scaled" not in st.session_state:
            st.session_state.X_test_scaled = []

    if "n_clusters" not in st.session_state:
        st.session_state.n_clusters = 4

    text = """Convolutional Neural Network Multi-class Image Classification"""
    st.subheader(text)

    text = """Prof. Louie F. Cervantes, M. Eng. (Information Engineering) \n
    CCS 229 - Intelligent Systems
    Department of Computer Science
    College of Information and Communications Technology
    West Visayas State University"""
    st.text(text)

    st.image('rock-paper-scissor.png', caption='Rock Paper Scissor')

    text = """
    This Streamlit app demonstrates a Convolutional Neural Network (CNN) for 
    rock-paper-scissors hand gesture classification, built using TensorFlow 
    Keras. It allows users to upload an image dataset and to see the trained model's prediction 
    for the hand sign (rock, paper, or scissors). Under the hood, the app 
    loads a pre-trained CNN model, preprocesses the uploaded image for the 
    model's input format, performs inference to generate class probabilities, 
    and displays the predicted class. This interactive environment provides computer 
    science students with a practical example of applying deep learning for image 
    classification tasks.
    """
    st.write(text)

    with st.expander("How to use this App"):
         text = """Step 1. Go to Training page. Set the parameters of the CNN. Click the button to begin training.
         \nStep 2.  Go to Performance Testing page and click the button to load the image
         and get the model's output on the classification task.
         \nYou can usw the training page to try other combinations of parameters."""
         st.write(text)
    
#run the app
if __name__ == "__main__":
    app()
