import concurrent.futures
import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import dlib
import numpy as np

import face_average

predictor_path = "shape_predictor_68_face_landmarks.dat"
faces_folder_path = "./imgs"
crops_folder_path = "./crops"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

files = glob.glob(os.path.join(faces_folder_path, "*.jp*g"))


def transform_warp_image(imgdata, eyecornerDst, boundaryPts, w, h, n, pointsNorm, imagesNorm, pointsAvg):
    points1 = imgdata["points"]
    # Corners of the eye in input image
    eyecornerSrc = [imgdata["points"][36],  points1[45]]
    for pt in imgdata["points"]:
        pass
        # cv2.circle(imgdata["img"], (pt[0], pt[1]),2,(0,0,255),2)
    # Compute similarity transform
    tform = face_average.similarityTransform(eyecornerSrc, eyecornerDst)

    # Apply similarity transformation
    img = cv2.warpAffine(imgdata["img"], tform, (w, h))

    # Apply similarity transform on points
    points2 = np.reshape(np.array(points1), (68, 1, 2))
    points = cv2.transform(points2, tform)
    points = np.float32(np.reshape(points, (68, 2)))

    # Append boundary points. Will be used in Delaunay Triangulation
    points = np.append(points, boundaryPts, axis=0)

    # Calculate location of average landmark points.
    pointsAvg = pointsAvg + points / n

    pointsNorm.append(points)
    imagesNorm.append(img)
    return pointsAvg

def process_pipeline(filename):
    process_image(filename)
    process_transform(filename)

    # now merge cropped images


def process_transform(filename):

    basename, file_extension = os.path.splitext(os.path.basename(filename))
    # Replace this with some proper merge order:
    allfiles = glob.glob(os.path.join(faces_folder_path, "*.jp*g"))

    imgidx = allfiles.index(filename)
    if (imgidx % 50 != 0 or imgidx == 0):
        return

    fileids = [os.path.splitext(os.path.basename(f))[0]
               for f in allfiles[imgidx: min(imgidx+10, len(allfiles))]]

    # magic happens here:
    w = 250
    h = 250
    filearr = face_average.read_points(
        crops_folder_path, faces_folder_path, fileids)

    # Eye corners
    eyecornerDst = [(np.int(0.3 * w), np.int(h / 3)),
                    (np.int(0.7 * w), np.int(h / 3))]

    imagesNorm = []
    pointsNorm = []

    # Add boundary points for delaunay triangulation
    boundaryPts = np.array(
        [(0, 0), (w/2, 0), (w-1, 0), (w-1, h/2), (w-1, h-1), (w/2, h-1), (0, h-1), (0, h/2)])

    # Initialize location of average points to 0s
    pointsAvg = np.array(
        [(0, 0)] * (len(filearr[0]["points"]) + len(boundaryPts)),
        np.float32())

    # Warp images and trasnform landmarks to output coordinate system,
    # and find average of transformed landmarks.
    for i in range(0, len(filearr)):
        pointsAvg = transform_warp_image(filearr[i], eyecornerDst, boundaryPts, w, h, len(filearr), pointsNorm, imagesNorm, pointsAvg)

    # Delaunay triangulation
    rect = (0, 0, w, h)
    dt = face_average.calculateDelaunayTriangles(rect, np.array(pointsAvg))

    # Output image
    output = np.zeros((h, w, 3), np.float32())

    # Warp input images to average image landmarks
    for i in range(0, len(imagesNorm)):
        img = np.zeros((h, w, 3), np.float32())
        # Transform triangles one by one
        for j in range(0, len(dt)):
            tin = []
            tout = []

            for k in range(0, 3):
                pIn = pointsNorm[i][dt[j][k]]
                pIn = face_average.constrainPoint(pIn, w, h)

                pOut = pointsAvg[dt[j][k]]
                pOut = face_average.constrainPoint(pOut, w, h)

                tin.append(pIn)
                tout.append(pOut)
            face_average.warpTriangle(imagesNorm[i], img, tin, tout)
        # Add image intensities for averaging
        output = output + img
    # Divide by numImages to get average
    output = output / len(filearr)

    # Display result
    cv2.imwrite("results/final{0}.png".format(basename), output)


def process_image(filename):
    basename, file_extension = os.path.splitext(os.path.basename(filename))
    # img = dlib.load_rgb_image(filename)
    imgcv = cv2.imread(filename, cv2.IMREAD_COLOR)
    img = imgcv

    dets = detector(img, 1)

    faces = []
    for k, d in enumerate(dets):
        # Get the landmarks/parts for the face in box d.
        crop = imgcv[d.top():d.bottom(), d.left():d.right()]
        cv2.imwrite("crops/{0}_f{1}.png".format(basename, k), crop)

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
        with open("crops/{0}_f{1}.json".format(basename, k), 'w') as outfile:
            json.dump(points, outfile)
        cv2.imwrite("crops/{0}_f{1}_o.png".format(basename, k), crop)

    cv2.imwrite("overlays/{0}.png".format(basename), imgcv)
    with open("overlays/{0}.json".format(basename), 'w') as outfile:
        json.dump(faces, outfile)


with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {executor.submit(process_pipeline, f): f for f in files[:400]}
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
