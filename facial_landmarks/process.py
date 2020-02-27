import sys
import os
import dlib
import glob
import cv2
import json
import numpy as np
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import multiprocessing



predictor_path = "shape_predictor_68_face_landmarks.dat"
faces_folder_path = "./imgs"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
# win = dlib.image_window()

files = glob.glob(os.path.join(faces_folder_path, "*.jp*g"))

def process_pipeline(filename):
    process_image(filename)

def process_image(filename):
    basename, file_extension = os.path.splitext(os.path.basename(filename))
    img = dlib.load_rgb_image(filename)
    imgcv = cv2.imread(filename,cv2.IMREAD_COLOR)

    dets = detector(img, 1)
    # print("Number of faces detected: {}".format(len(dets)))
    
    faces = []
    for k, d in enumerate(dets):
        # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format( k, d.left(), d.top(), d.right(), d.bottom()))

        # Get the landmarks/parts for the face in box d.
        crop = imgcv[d.top():d.bottom(), d.left():d.right()]

        cv2.imwrite("crops/{0}_f{1}.png".format(basename, k), crop)

        # predict ones for the entire image
        shape = predictor(img, d)
        face_crop_dets = detector(img, 1)
        face_crop_shape = predictor(crop, d)
        points = []
        face_crop_points = []
        for i in range(shape.num_parts): 
            cv2.circle(imgcv, (shape.part(i).x, shape.part(i).y),2,(0,0,255),2)
            cv2.circle(crop, (face_crop_shape.part(i).x, face_crop_shape.part(i).y),2,(0,0,255),2)
            points.append( (shape.part(i).x, shape.part(i).y) )
            face_crop_points.append( (face_crop_shape.part(i).x, face_crop_shape.part(i).y) )
        
        faces.append({"box": [d.top(), d.bottom(), d.left(), d.right()], "points": points })
        # write the points calculated based on the cropped image.
        with open("crops/{0}_f{1}.json".format(basename, k), 'w') as outfile:
            json.dump(face_crop_points, outfile)
        cv2.imwrite("crops/{0}_f{1}_o.png".format(basename, k), crop)

    cv2.imwrite("overlays/{0}.png".format(basename), imgcv) 
    with open("overlays/{0}.json".format(basename), 'w') as outfile:
            json.dump(faces, outfile)
    # dlib.hit_enter_to_continue()

with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
    futures = { executor.submit(process_pipeline, f): f for f in files[:100] }
    for future in concurrent.futures.as_completed(futures):
        result = futures[future]
        try:
            result = future.result()
        except Exception as e:
            print('%r generated an exception: %s' % (result, e))
        else:
            # do something
            pass
print("finished files")
