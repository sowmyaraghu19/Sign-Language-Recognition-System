# Sign-Language-Recognition-System
A computer vision-based project designed to recognize and interpret Indian Sign Language gestures using a convolutional neural network (CNN). This system enhances accessibility for individuals with hearing or speech impairments by providing gesture recognition capabilities, word suggestions, and audio-to-text conversion.

# Features
1. Image Preprocessing: Converts gesture images into a format suitable for model training.
2. Dataset Collection: Captures and organizes gesture images for each letter of the alphabet.
3. Sign Recognition: Uses a CNN to predict the gesture's corresponding alphabet or word.
4. Word Suggestions: Implements Hunspell to suggest words when gesturing letter-by-letter.
5. Audio-to-Text Conversion: Converts recognized text into audio in various languages using gTTS.

# Project Files
1. image_processing.ipynb:
Performs image preprocessing, including grayscale conversion, Gaussian blur, and adaptive thresholding, to prepare images for training.

2. datacollection.ipynb:
Collects gesture data from the user via webcam and organizes the dataset into training and testing directories for each alphabet (A-Z).

# Prerequisites
Python 3.8 or above
OpenCV
NumPy
Hunspell
gTTS
TensorFlow/Keras (for model implementation)
Webcam for data collection

# How to Run
Step 1: Data Collection:
Run datacollection.ipynb to capture gestures for each alphabet.
Use the webcam interface to record your gestures and save them into the appropriate dataset folder.

Step 2: Preprocessing:
Use image_processing.ipynb to preprocess the captured images, converting them into grayscale and applying adaptive thresholding for consistency.

Step 3: Model Training (Optional):
Train your CNN model using the processed dataset or use a pre-trained model (to be added in future iterations).

Step 4: Testing:
Use the model to predict gestures in real time and generate corresponding letters or words.

Step 5: Additional Features:
Activate Hunspell for word suggestions and gTTS for converting text to speech.

# Technologies Used
OpenCV: For image processing and webcam data capture.
NumPy: For numerical computations.
Hunspell: For word suggestion.
gTTS: For text-to-speech conversion.
TensorFlow/Keras: For building and training the CNN.

# Acknowledgments
Indian Sign Language Dataset: Inspired by commonly used Indian Sign Language gestures.
OpenCV Documentation: For guidance on real-time image processing.
Hunspell and gTTS: For enhancing accessibility features.
This project aims to bridge the communication gap between individuals with hearing or speech impairments and the rest of the world, making technology more inclusive.

Feel free to contribute or suggest enhancements! 😊
