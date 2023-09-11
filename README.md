# Drowsiness Detection Model

This repository contains Python code for a Drowsiness Detection System using computer vision techniques and a Convolutional Neural Network (CNN) model. The system is designed to detect whether a person's eyes are open or closed and trigger an alarm when drowsiness is detected.

## Code Explanation

### drowsiness_detection.py

This is the main script responsible for capturing video from the camera, detecting faces and eyes, classifying eye states (open or closed), and triggering an alarm. The code can be summarized as follows:

- **Importing Libraries**: Imports necessary libraries, including OpenCV, Keras, and Pygame.
- **Initialize Mixer and Load Resources**: Initializes Pygame for sound playback and loads an alarm sound.
- **Load Haar Cascade Classifiers**: Loads Haar cascade classifiers for face, left eye, and right eye detection.
- **Define Labels and Load CNN Model**: Defines labels for eye states ('Close' and 'Open') and loads a pre-trained CNN model for eye state classification.
- **Capture Video**: Captures video from the default camera (usually the webcam).
- **Main Loop**: Processes each frame from the camera feed and performs the following tasks:
  - Detects faces, left eyes, and right eyes using Haar cascades.
  - Classifies the eye state for each eye using the loaded CNN model.
  - Calculates a "score" based on the number of consecutive closed-eye predictions.
  - Displays the score and whether the eyes are closed or open on the frame.
  - Triggers an alarm when the score exceeds a threshold.
- **Exit the Loop**: Terminates the loop and releases the camera when the 'q' key is pressed.

### model.py

This script is used to build, train, and save a CNN model for the binary image classification task of detecting open and closed eyes. The key components of the script are as follows:

- **Importing Libraries**: Imports necessary libraries for image processing and deep learning.
- **Generator Function**: Defines a function to create a data generator for image data, specifying options such as data augmentation, batch size, target image size, and class mode.
- **Data Preparation**: Sets batch size and target image size and creates data generators for training and validation data.
- **Model Architecture**: Defines a CNN model using the Keras Sequential API, including convolutional, max-pooling, dropout, flatten, and fully connected layers.
- **Model Compilation**: Compiles the model with the Adam optimizer, categorical cross-entropy loss function, and accuracy as the evaluation metric.
- **Model Training**: Trains the model using the training and validation data generators, specifying the number of training epochs and steps per epoch.
- **Model Saving**: Saves the trained model as 'cnnCat2.h5' in the 'models' directory.

## Mathematical Equations

### Scoring Mechanism (Used in drowsiness_detection.py)

- Let `score` be the drowsiness score.
- For each frame, if both eyes are classified as closed, increment the score: `score += 1`.
- If at least one eye is classified as open, decrement the score: `score -= 1`.
- The alarm is triggered when the score exceeds a certain threshold (`score > 7`).

## Synchronization of Files

- `drowsiness_detection.py` is responsible for real-time drowsiness detection using the trained model.
- `model.py` is responsible for building, training, and saving the CNN model used for eye state classification.

## Screenshots and Output

Include screenshots or describe the expected output of your Drowsiness Detection System here.

## Usage

1. Ensure you have the required libraries installed, including OpenCV, Keras, and Pygame.
2. Prepare labeled training and validation datasets, placing them in the 'data/train' and 'data/valid' directories.
3. Run `model.py` to build and train the CNN model.
4. Run `drowsiness_detection.py` to use the trained model for real-time drowsiness detection.

## Dependencies

- OpenCV
- Keras
- numpy
- Pygame
