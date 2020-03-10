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


def transform_warp_image_only(fimg, imgpoints, eyecornerDst, w, h):
    # Corners of the eye in input image
    eyecornerSrc = [imgpoints[36], imgpoints[45]]
    # Compute similarity transform
    tform = face_average.similarityTransform(eyecornerSrc, eyecornerDst)
    # Apply similarity transformation
    img = cv2.warpAffine(fimg, tform, (w, h))
    return img

def transform_landmarks_only(imgpoints, eyecornerDst, boundaryPts, n, pointsAvg):
    points1 = imgpoints

    # Corners of the eye in input image
    eyecornerSrc = [imgpoints[36], points1[45]]

    # Compute similarity transform
    tform = face_average.similarityTransform(eyecornerSrc, eyecornerDst)

    # Apply similarity transform on points
    points2 = np.reshape(np.array(points1), (68, 1, 2))
    points = cv2.transform(points2, tform)
    points = np.float32(np.reshape(points, (68, 2)))

    # Append boundary points. Will be used in Delaunay Triangulation
    points = np.append(points, boundaryPts, axis=0)

    # Calculate location of average landmark points.
    pointsAvg = pointsAvg + points / n
    return (pointsAvg , points)


def process_transform(ids, grpname, facesdf, ofname):
    faces = facesdf[facesdf.imgid.isin(ids)]
    faces_length = faces.shape[0]
    imgs = {}
    for i, row in faces.iterrows():
        if row.imgid not in imgs.keys():
            imgs[row.imgid] = "keep"

    if faces.shape[0] < 2:
        return
    # magic happens here:
    w = 1000
    h = 1000
    pointf = max(int((w / 250)), 1)

    # Eye corners
    eyecornerDst = [(np.int(0.3 * w), np.int(h / 3)),
                    (np.int(0.7 * w), np.int(h / 3))]

    pointsNorm = []

    # Add boundary points for delaunay triangulation
    boundaryPts = np.array(
        [(0, 0), (w/2, 0), (w-1, 0), (w-1, h/2), (w-1, h-1), (w/2, h-1), (0, h-1), (0, h/2)])

    # Initialize location of average points to 0s
    pointsAvg = np.array(
        [(0, 0)] * (len(faces.iloc[0]["points"]) + len(boundaryPts)),
        np.float32())

    # Warp images and trasnform landmarks to output coordinate system,
    # and find average of transformed landmarks.
    for i in range(0, faces.shape[0]):
        pointsAvg, normedPoints = transform_landmarks_only(faces.iloc[i].points, eyecornerDst, boundaryPts, faces_length, pointsAvg)
        pointsNorm.append(normedPoints)

    # Delaunay triangulation
    rect = (0, 0, w, h)
    dt = face_average.calculateDelaunayTriangles(rect, np.array(pointsAvg))

    # Output image
    output = np.zeros((h, w, 3), np.float32())

    # Warp input images to average image landmarks
    for i in range(0, faces_length):
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
            
            sourceimg = cv2.imread(os.path.join("imgs", "{fid}.jpg".format(fid=faces.iloc[i].imgid))) 
            normedimg = transform_warp_image_only(sourceimg, faces.iloc[i].points, eyecornerDst, w, h)
            face_average.warpTriangle(normedimg, img, tin, tout)
            # cv2.imwrite("debug/{0}_{1}_.jpg".format(str(grpname), i), sourceimg)
            # cv2.imwrite("debug/{0}_{1}.jpg".format(str(grpname), i), normedimg)
        # Add image intensities for averaging
        # cv2.imwrite("debug/2_{0}_{1}.jpg".format(str(grpname), i), imagesNorm[i])
        output = output + img
    # Divide by numImages to get average
    output = output / faces.shape[0]

    # Get the average points of the face
    averageFacialLandmarks = pointsAvg[:-(len(boundaryPts))]
    averageFacialLandmarks = [ (int(pt[0]), int(pt[1])) for pt in averageFacialLandmarks]

    r, g, b = face_average.extract_dominant_color(
        output, averageFacialLandmarks)

    # Plot a mask
    facemaskimg = np.zeros((h, w, 4), np.float32())
    for pt in averageFacialLandmarks:
        cv2.circle(facemaskimg, (int(pt[0]), int(
            pt[1])), 3, (0, 0, 0, 255), pointf * 3)
        cv2.circle(facemaskimg, (int(pt[0]), int(
            pt[1])), 2, (b, g, r, 255), pointf * 2)

    # Display result
    cv2.imwrite(
        "results/{1}/{0}_mask.png".format(str(grpname), ofname), facemaskimg)
    cv2.imwrite("results/{1}/{0}.jpg".format(str(grpname), ofname), output)

    return {"groupkey": grpname, "images": list(imgs.keys()), "faces": [], "landmarks": averageFacialLandmarks}


def process_row(grpdata, faces_df, ofname):
    grpname, grp = grpdata
    ids = list()
    if (isinstance(grpname, tuple)):
        grpname = "-".join(map(str, grpname))
    else:
        grpname = str(grpname)

    for row_index, row in grp.iterrows():
       ids.append(row_index)
    return process_transform(ids, grpname, faces_df, ofname)


def process_dataframe(ofname, grouped_df, face_df):
    rdir = os.path.join("results", ofname)
    ddir = os.path.join("debug", ofname)
    if not os.path.exists(rdir):
        os.makedirs(rdir)
    if not os.path.exists(ddir):
        os.makedirs(ddir)
    with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        no_downloaded = 0
        futures = {executor.submit(
            process_row, grp, face_df, ofname): grp for grp in grouped_df}
        face_warp_hist = {}
        for future in concurrent.futures.as_completed(futures):
            result = futures[future]
            try:
                result = future.result()
                if (isinstance(result, dict)):
                    face_warp_hist[result["groupkey"]] = {
                        "images": result["images"],
                        "landmarks": result["landmarks"]
                        }
            except Exception as e:
                print('%r generated an exception: %s' % (result, e))
            else:
                no_downloaded += 1
        print("finished files {0}".format(no_downloaded))
        with open("results/{0}.json".format(ofname), 'w') as outfile:
            json.dump(face_warp_hist, outfile)


faces_df = pd.read_json("faces.json")

omniart_df = pd.read_csv("omniart_v3_portrait.csv", encoding='utf8')

omniart_df.creation_year = pd.to_numeric(
    omniart_df.creation_year, downcast='integer')
omniart_df = omniart_df[omniart_df.id.isin(faces_df.imgid.unique())]

omnifaces_df = faces_df.set_index("imgid").join(omniart_df.set_index("id"))

# Warp images depending on group
omnifaces_df.sort_values(by="creation_year")
omnifaces_grouped = omnifaces_df.groupby(by=["creation_year"])
process_dataframe("yearly", omnifaces_grouped, faces_df)
print("yearly done")

omnifaces_df.sort_values(by="creation_year")
omnifaces_grouped = omnifaces_df.groupby(by=["gender", "creation_year"])
process_dataframe("yearly-gender", omnifaces_grouped, faces_df)
print("yearly gender done")

omnifaces_df.sort_values(by="creation_year")
omnifaces_grouped = omnifaces_df.groupby(by=["age", "creation_year"])
process_dataframe("yearly-age", omnifaces_grouped, faces_df)
print("yearly age done")

omnifaces_df["decade"] = omnifaces_df.creation_year.floordiv(10)
omnifaces_df.sort_values(by="decade")
omnifaces_grouped = omnifaces_df.groupby(by=["decade"])
process_dataframe("decade", omnifaces_grouped, faces_df)
print("decade done")

omnifaces_grouped = omnifaces_df.groupby(by=["gender", "decade"])
process_dataframe("decade-gender", omnifaces_grouped, faces_df)
print("decade gender done")

omnifaces_grouped = omnifaces_df.groupby(by=["age", "decade"])
process_dataframe("decade-age", omnifaces_grouped, faces_df)
print("decade age done")

omnifaces_df["century"] = omnifaces_df.creation_year.floordiv(100)
omnifaces_df.sort_values(by="century")
omnifaces_grouped = omnifaces_df.groupby(by=["century"])
process_dataframe("century", omnifaces_grouped, faces_df)
print("century done")

omnifaces_grouped = omnifaces_df.groupby(by=["gender", "century"])
process_dataframe("century-gender", omnifaces_grouped, faces_df)
print("century gender done")

omnifaces_grouped = omnifaces_df.groupby(by=["age", "century"])
process_dataframe("century-age", omnifaces_grouped, faces_df)
print("century age done")