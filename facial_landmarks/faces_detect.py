import concurrent.futures
import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import dlib
import numpy as np
import multiprocessing

import face_average

def process_image(filename):
    basename, file_extension = os.path.splitext(os.path.basename(filename))
    # img = dlib.load_rgb_image(filename)
    imgcv = cv2.imread(filename, cv2.IMREAD_COLOR)
    img = imgcv.copy()

    dets = detector(img, 1)

    faces = []
    for k, d in enumerate(dets):
        # Get the landmarks/parts for the face in box d.
        crop = imgcv[d.top():d.bottom(), d.left():d.right()]
        # cv2.imwrite("crops/{0}_fr{1}.png".format(basename, k), crop)

        # predict ones for the entire image
        shape = predictor(img, d)
        points = []
        for i in range(shape.num_parts):
            cv2.circle(imgcv, (shape.part(i).x, shape.part(i).y),
                       2, (0, 0, 255), 2)
            raw_points = (shape.part(i).x, shape.part(i).y)
            points.append(raw_points)

        faces.append({
            "box": [d.top(), d.bottom(), d.left(), d.right()],
            "points": points
        })
        # write the points calculated based on the cropped image.
        # with open("crops/{0}_fd{1}.json".format(basename, k), 'w') as outfile: json.dump(points, outfile)
        # cv2.imwrite("crops/{0}_fp{1}.png".format(basename, k), crop)
    if (len(faces) > 0):
        # cv2.imwrite("overlays/{0}_an.png".format(basename), imgcv)
        # cv2.imwrite("overlays/{0}.png".format(basename), img)
        with open("overlays/{0}.json".format(basename), 'w') as outfile:
            json.dump(faces, outfile)
    else:
        cv2.imwrite("missed/{0}.png".format(basename), img)

def process_pipeline(filename):
    process_image(filename)

faces_folder_path = "./imgs"
crops_folder_path = "./crops"
predictor_path = "shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

files = glob.glob(os.path.join(faces_folder_path, "*.*"))

print("processing {0} files".format(len(files)))

with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
    futures = {executor.submit(process_pipeline, f): f for f in files}
    for future in concurrent.futures.as_completed(futures):
        result = futures[future]
        try:
            result = future.result()
        except Exception as e:
            print('%r generated an exception: %s' % (result, str(e)))
        else:
            # do something
            pass
print("finished files")
