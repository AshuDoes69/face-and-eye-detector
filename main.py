import cv2
import matplotlib.pyplot as plt

# setting up the face and eye detectors
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
#in the above line, we are using the pre-trained models from OpenCV
# making a list of image file names
#now are importing the images from the images folder

import os
p = os.listdir('/images')
q = []
for i in p:
    if ".jpg" in i:
        q.append(i)
#here we are making a list of all the images in the images folder
def detect_face(x):
    img = cv2.imread(f'/images/{x}')
#this is the function that will detect the face and eyes in the image
    # Detect faces
    faces = face_cascade.detectMultiScale(img, 1.1, 4)
    eyes  = eye_cascade.detectMultiScale(img, 1.1, 4)

    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 5)
    for (x, y, w, h) in eyes:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
    # Convert images into RGB
    im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Display the output
    plt.imshow(im_rgb)
    plt.show()

# here we are calling the function for the first 10 images
for i in q[106:110]:
    detect_face(i)
    