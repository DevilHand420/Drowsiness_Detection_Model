import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

# Initialize Pygame mixer for sound
mixer.init()
# Load the alarm sound
sound = mixer.Sound('alarm.wav')

# Load Haar cascade classifiers for face and eyes
face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

# Labels for eye states (open or closed)
lbl = ['Close', 'Open']

# Load the pre-trained CNN model for eye state classification
model = load_model('models/cnncat2.h5')
path = os.getcwd()

# Open the default camera (usually the webcam)
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
score = 0
thicc = 2
rpred = [99]
lpred = [99]

while(True):
    ret, frame = cap.read()
    height, width = frame.shape[:2] 

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    # Draw a filled rectangle at the bottom of the frame for displaying text
    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    # Loop over detected faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    # Loop over detected right eyes
    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y + h, x:x + w]
        count += 1
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (24, 24))
        r_eye = r_eye / 255
        r_eye = r_eye.reshape(24, 24, -1)
        r_eye = np.expand_dims(r_eye, axis=0)
        rpred_temp = model.predict(r_eye)
        rpred = np.argmax(rpred_temp)
        print(rpred)
        if rpred == 1:
            lbl = 'Open' 
        if rpred == 0:
            lbl = 'Closed'
        break

    # Loop over detected left eyes
    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y + h, x:x + w]
        count += 1
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (24, 24))
        l_eye = l_eye / 255
        l_eye = l_eye.reshape(24, 24, -1)
        l_eye = np.expand_dims(l_eye, axis=0)
        lpred_temp = model.predict(l_eye)
        lpred = np.argmax(lpred_temp)
        print(lpred)
        if lpred == 1:
            lbl = 'Open'
        if lpred == 0:
            lbl = 'Closed'
        break

    # Calculate the drowsiness score based on eye states
    if rpred == 0 and lpred == 0:
        score += 1
        cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        score -= 1
        cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # Ensure the score is non-negative
    if score < 0:
        score = 0

    # Display the current score on the frame
    cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # Stop the alarm sound if the score is less than 5
    if score < 5:
        sound.stop()

    # Trigger the alarm if the score exceeds 7
    if score > 7:
        # Save a snapshot of the frame
        cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
        try:
            sound.play()
        except:
            pass

        # Adjust the thickness of the red border
        if thicc < 16:
            thicc = thicc + 2
        else:
            thicc = thicc - 2
            if thicc < 2:
                thicc = 2
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

    # Display the frame
    cv2.imshow('frame', frame)

    # Exit the loop when the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
