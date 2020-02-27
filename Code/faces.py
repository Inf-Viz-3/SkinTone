import face_recognition
from PIL import Image
import pandas as pd
import imutils
import urllib
import cv2
import numpy as np
from PIL import Image



data = pd.read_excel(r'D:/VU/Information_Visualization/untitled/met.xlsx')
df = pd.DataFrame(data, columns= ['id', 'image_url'])
dict2=[]

#
for i in range(10671, len(df['id'])):
    # img = imutils.url_to_image(df['image_url'])
    # image1 = cv2.imread(img, cv2.COLOR_BGR2RGB)

    resp = urllib.request.urlopen(df['image_url'][i])
    arr = np.asarray(bytearray(resp.read()), dtype=np.uint8)
    img= cv2.imdecode(arr, -1)
    if (len(img.shape) < 3):
        image1 = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        image = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    elif len(img.shape) == 3:
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        image = cv2.cvtColor(img, cv2.COLOR_RGB2BRG)
    # image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    face_locations = face_recognition.face_locations(image)
    print("Found {} face(s) in a photo.".format(len(face_locations)), " Photo: {}".format(df['id'][i]))

    if len(face_locations)==0:
        dict2.append(df['id'][i])

    for face_location in face_locations:

        # Print the location of each face in this image
        top, right, bottom, left = face_location
        print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

        # You can access the actual face itself like this:
        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        pil_image.save('D:/VU/Information_Visualization/Faces/{}.jpg'.format(df['id'][i]), 'JPEG')




# img = cv2.imread('D:/VU/Information_Visualization/Faces/632d08c7-db0f-43c1-ab55-9ef92b591084.jpg', cv2.IMREAD_COLOR)
# image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# pil_image = Image.fromarray(image)
# pil_image.save('D:/VU/Information_Visualization/Faces/5.jpg', 'JPEG')