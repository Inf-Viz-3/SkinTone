import concurrent.futures
import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor
import imutils
import cv2
import dlib
import numpy as np
import pandas as pd
import multiprocessing
from skin import extractDominantColor, extractSkin
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
import face_average

def resizewithratio(img):
    height, width = img.shape[:2]
    max_height = 250
    max_width = 250
    
    # get scaling factor
    scaling_factor = max_height / float(height)
    if max_width/float(width) < scaling_factor:
        scaling_factor = max_width / float(width)
    # resize image
    img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

    return img

def get_age_gender(face):
    ageProto = "age_deploy.prototxt"
    ageModel = "age_net.caffemodel"

    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
               '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['male', 'female']

    blob = cv2.dnn.blobFromImage(
        face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    genderNet.setInput(blob)
    genderPreds = genderNet.forward()
    gender = genderList[genderPreds[0].argmax()]

    ageNet.setInput(blob)
    agePreds = ageNet.forward()
    age = ageList[agePreds[0].argmax()]
    
    return gender, age


def process_image(filename):
    basename, file_extension = os.path.splitext(os.path.basename(filename))

    imgcv = cv2.imread(filename, cv2.IMREAD_COLOR)

    dets = detector(imgcv, 1)

    if (len(dets) == 0):
        cv2.imwrite("missed/{0}.jpg".format(basename), imgcv)
        return None

    imgfaces_df = pd.DataFrame()

    for k, d in enumerate(dets):
        # predict facelandmarks for the entire image
        shape = predictor(imgcv, d)
        points = []

        # extract the main face to determine color (points 1, 28, 17, 9)
        facecrop = imgcv[d.top():d.bottom(), d.left():d.right()]

        face_gender, face_age = get_age_gender(imgcv)
        r, g, b = face_average.extract_dominant_color(facecrop)

        for i in range(shape.num_parts):
            raw_points = (shape.part(i).x, shape.part(i).y)
            points.append(raw_points)

        result = {
            'imgid': [basename],
            'faceid': [k],
            'box': [[d.top(), d.bottom(), d.left(), d.right()]],
            'points': [points],
            "gender": face_gender,
            "age": face_age,
            "color": [(r, g, b)]
        }
        df = pd.DataFrame(data=result)

        cv2.imwrite("faces/{0}_{1}.jpg".format(basename, k), resizewithratio(facecrop))

        # Paint dominant color rect
        cv2.rectangle(imgcv, (d.left(), d.top()), (d.right(), d.bottom()), (b, g, r), 5)

        imgfaces_df = imgfaces_df.append(df)

    cv2.imwrite("overlays/{0}.jpg".format(basename), imgcv)

    return imgfaces_df


def process_pipeline(filename):
    return process_image(filename)

faces_folder_path = "./imgs"
crops_folder_path = "./crops"
predictor_path = "shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

files = glob.glob(os.path.join(faces_folder_path, "*.*"))

print("processing {0} files".format(len(files)))
faces_df = pd.DataFrame()

with ThreadPoolExecutor(max_workers=1) as executor:
    futures = {executor.submit(process_pipeline, f): f for f in files}
    for future in concurrent.futures.as_completed(futures):
        result = futures[future]
        try:
            result = future.result()
            if (result is not None):
                faces_df = faces_df.append(result)
        except Exception as e:
            print('%r generated an exception: %s' % (result, str(e)))
        else:
            # do something
            pass

# Group colors

number_of_colors = 9
colors = pd.DataFrame([], columns=['R', 'G', 'B', 'imgid', 'faceid'])
for idx, rowobj in enumerate(faces_df.iterrows()):
    row = rowobj[1]
    r = row['color'][0]
    g = row['color'][1]
    b = row['color'][2]
    imgid = row['imgid']
    faceid = row['faceid']
    colors.loc[idx] = [r, g, b, imgid, faceid]
    print(idx)

print(colors)

estimator = KMeans(n_clusters=number_of_colors, random_state=0)

# Fit the image
estimator.fit(colors[['R', 'G', 'B']])

colorsgrouped = pd.DataFrame(
    {'group': estimator.labels_, 'R': colors['R'], 'G': colors['G'], 'B': colors['B'], 'imgid': colors['imgid'], 'faceid': colors['faceid']})

new_faces = pd.merge(faces_df, colorsgrouped[['group', 'imgid', "faceid"]], on=[
                     'imgid', "faceid"], how='outer')

new_faces.to_json("faces.json", orient="records")
print("finished faces", faces_df.shape[0])
print("finished new faces", new_faces.shape[0])
