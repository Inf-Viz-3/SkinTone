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

faces_folder_path = "./imgs"
crops_folder_path = "./crops"

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

def process_transform(ids, year):

    filearr = face_average.read_points("overlays", ids)

    if len(filearr) == 0:
        return
    # magic happens here:
    w = 250
    h = 250
    
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
    cv2.imwrite("results/final{0}_w{1}.png".format(int(year), len(filearr)), output)


omniart_df = pd.read_csv("omniart_v3_portrait.csv", encoding = 'utf8')
omniart_df = omniart_df.head(1000)
omniart_df.sort_values(by="creation_year")
omniart_by_year_grp = omniart_df.groupby(by="creation_year")

def process_row(grpdata):
    name, grp = grpdata
    ids = list()
    for row_index, row in grp.iterrows():
        ids.append(row.id)
    process_transform(ids, name)

with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
    no_downloaded = 0
    futures = { executor.submit(process_row, grp,): grp for grp in omniart_by_year_grp }
    for future in concurrent.futures.as_completed(futures):
        result = futures[future]
        try:
          result = future.result()
        except Exception as e:
            print('%r generated an exception: %s' % (result, e))
        else:
            no_downloaded+=1
    print("finished files {0}".format(no_downloaded))