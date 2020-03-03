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

import face_average

def process_image(filename):
    basename, file_extension = os.path.splitext(os.path.basename(filename))
    
    imgcv = cv2.imread(filename, cv2.IMREAD_COLOR)

    dets = detector(imgcv, 1)
    imgfaces_df = pd.DataFrame()
    for k, d in enumerate(dets):
        # Get the landmarks/parts for the face in box d.
        crop = imgcv[d.top():d.bottom(), d.left():d.right()]

        # predict ones for the entire image
        shape = predictor(imgcv, d)
        points = []
        for i in range(shape.num_parts):
            raw_points = (shape.part(i).x, shape.part(i).y)
            points.append(raw_points)
        df = pd.DataFrame(data={
            'imgid': [basename],
            'faceid': [k],
            'box': [[d.top(), d.bottom(),d.left(), d.right()]],
            'points': [points]
        })
        imgfaces_df = imgfaces_df.append(df)
    
        
    if (imgfaces_df.count == 0):
        cv2.imwrite("missed/{0}.png".format(basename), imgcv)
        return None
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
    futures = {executor.submit(process_pipeline, f): f for f in files[:100]}
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
faces_df = faces_df.set_index(["imgid", "faceid"])
faces_df.to_csv("faces.csv")
print("finished files")
