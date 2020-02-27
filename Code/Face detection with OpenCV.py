# Import the necessary libraries
import numpy as np
import cv2 
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')

#
def detect_faces(cascade, test_image, scaleFactor = 1.1):
    # create a copy of the image to prevent any changes to the original one.
    image_copy = test_image.copy()

    #convert the test image to gray scale as opencv face detector expects gray images
    gray_image = cv2.cvtColor(image_copy, cv2.COLOR_BGR2GRAY)

    # Applying the haar classifier to detect faces
    faces_rect = cascade.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors = 5)

    for (x, y, w, h) in faces_rect:
        cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return image_copy


def convertToRGB(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Testing the function on new image


#loading image
test_image2 = cv2.imread('D:/VU/Information_Visualization/untitled/data/baby1.png')
haar_cascade_face = cv2.CascadeClassifier('D:/VU/Information_Visualization/untitled/data/haarcascades/haarcascade_frontalface_alt2.xml')
#call the function to detect faces
faces = detect_faces(haar_cascade_face, test_image2)

#convert to RGB and display image
plt.imshow(convertToRGB(faces))

# Saving the final image

cv2.imwrite('D:/VU/Information_Visualization/untitled/data/777.png', faces)
