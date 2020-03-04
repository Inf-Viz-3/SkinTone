import concurrent.futures
import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import dlib
import numpy as np
import pandas as pd
import multiprocessing
from skin import extractDominantColor

import face_average

def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, bboxes

def scan(frame):

    # Load network
    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"

    ageProto = "age_deploy.prototxt"
    ageModel = "age_net.caffemodel"

    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)
    faceNet = cv2.dnn.readNet(faceModel, faceProto)

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']
    
    padding = 20

    frameFace, bboxes = getFaceBox(faceNet, frame)
    if not bboxes:
        # print(f"\t No face Detected, Checking next frame")
        return [], []

    gender_ = []
    age_ = []
    for bbox in bboxes:
        #print(f'\t Face Detected')
        face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        gender_.append(gender)
        
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        age_.append(age)

    return gender_, age_


from sklearn.cluster import KMeans

def process_image(filename):
    basename, file_extension = os.path.splitext(os.path.basename(filename))
    
    imgcv = cv2.imread(filename, cv2.IMREAD_COLOR)

    dets = detector(imgcv, 1)

    
    if (len(dets) == 0):
        cv2.imwrite("missed/{0}.png".format(basename), imgcv)
        return None

    imgfaces_df = pd.DataFrame()
    faces_gender, faces_age = scan(imgcv)

    if len(faces_gender) != len(dets):
        return None

    for k, d in enumerate(dets):
        # predict facelandmarks for the entire image
        shape = predictor(imgcv, d)
        points = []

        # extract the main face to determine color (points 1, 28, 17, 9)
        try:
            crop = imgcv[d.top():d.bottom(), d.left():d.right()]
            skin = extractSkin(crop)
            color = extractDominantColor(skin)
            dominant_color = color[0].get("color")
            # TODO: if each of this is smaller than than, take other 
        except:
            crop = imgcv[shape.part(27).y:shape.part(8).y, shape.part(0).x:shape.part(16).x]
            dominant_color = crop.mean(axis=0).mean(axis=0)

        for i in range(shape.num_parts):
            raw_points = (shape.part(i).x, shape.part(i).y)
            points.append(raw_points)
        df = pd.DataFrame(data={
            'imgid': [basename],
            'faceid': [k],
            'box': [[d.top(), d.bottom(),d.left(), d.right()]],
            'points': [points],
            "gender": faces_gender[k],
            "age": faces_age[k],
            "color": [dominant_color]
        })
        imgfaces_df = imgfaces_df.append(df)

        
        cv2.rectangle(imgcv,
            (shape.part(0).x, shape.part(27).y),
            (shape.part(16).x, shape.part(8).y), (dominant_color[0], dominant_color[1],dominant_color[2]), 5)

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

with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
    futures = {executor.submit(process_pipeline, f): f for f in files}
    for future in concurrent.futures.as_completed(futures):
        result = futures[future]
        try:
            result = future.result()
            if (result is not None):
                pass
                faces_df = faces_df.append(result)
        except Exception as e:
            print('%r generated an exception: %s' % (result, str(e)))
        else:
            # do something
            pass
faces_df.to_json("faces.json", orient="records")
print("finished files")
