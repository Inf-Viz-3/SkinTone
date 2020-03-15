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
    return (pointsAvg, points)

def calculate_distance(x, y):
            sumval = ((x[0]-y[0])**2) + ((x[1]-y[1])**2)
            distance = (sumval)**0.5
            return distance

def process_transform(ids, grpname, facesdf, ofname, memimgs):
    faces = facesdf[facesdf.imgid.isin(ids)]
    faces_length = faces.shape[0]
    imgfaces = {}
    distances = {}

    for i, row in faces.iterrows():
        if row.imgid not in imgfaces.keys():
            imgfaces[row.imgid] = [row.faceid]
        else:
            imgfaces[row.imgid].append(row.faceid)

    if faces.shape[0] < 2:
        return
    # magic happens here:
    w = 1000
    h = 1000
    pointf = max(int((w / 250)), 1)

    # Eye corners
    eyecornerDst = [(np.int(0.3 * w), np.int(h / 3)),
                    (np.int(0.7 * w), np.int(h / 3))]

    pointsNorm = {}

    # Add boundary points for delaunay triangulation
    boundaryPts = np.array(
        [(0, 0), (w/2, 0), (w-1, 0), (w-1, h/2), (w-1, h-1), (w/2, h-1), (0, h-1), (0, h/2)])

    # Initialize location of average points to 0s
    pointsAvg = np.array(
        [(0, 0)] * (len(faces.iloc[0]["points"]) + len(boundaryPts)),
        np.float32())

    # Warp images and trasnform landmarks to output coordinate system,
    # and find average of transformed landmarks.
    for i, row in faces.iterrows():
        pointsAvg, normedPoints = transform_landmarks_only(
            row.points, eyecornerDst, boundaryPts, faces_length, pointsAvg)
        pointsNorm[f"{row.imgid}-{row.faceid}"] = normedPoints

    # Delaunay triangulation
    rect = (0, 0, w, h)
    dt = face_average.calculateDelaunayTriangles(rect, np.array(pointsAvg))

    # Get the average points of the face
    averageFacialLandmarks = pointsAvg[:-(len(boundaryPts))]
    averageFacialLandmarks = [(int(pt[0]), int(pt[1]))
                              for pt in averageFacialLandmarks]

    # Output image
    output = np.zeros((h, w, 3), np.float32())

    # Warp input images to average image landmarks
    result = {}
    for i, row in faces.iterrows():
        img = np.zeros((h, w, 3), np.float32())

        # Transform triangles one by one
        for j in range(0, len(dt)):
            tin = []
            tout = []
            for k in range(0, 3):
                pIn = pointsNorm[f"{row.imgid}-{row.faceid}"][dt[j][k]]
                pIn = face_average.constrainPoint(pIn, w, h)
                pOut = pointsAvg[dt[j][k]]
                pOut = face_average.constrainPoint(pOut, w, h)

                tin.append(pIn)
                tout.append(pOut)

            sourceimg = memimgs[row.imgid]
            normedimg = transform_warp_image_only(
                sourceimg, row.points, eyecornerDst, w, h)
            face_average.warpTriangle(normedimg, img, tin, tout)
            # cv2.imwrite("debug/{0}_{1}_.jpg".format(str(grpname), i), sourceimg)
            # cv2.imwrite("debug/{0}_{1}.jpg".format(str(grpname), i), normedimg)
        # Add image intensities for averaging
        # cv2.imwrite("debug/2_{0}_{1}.jpg".format(str(grpname), i), imagesNorm[i])
        faceLandMarks = pointsNorm[f"{row.imgid}-{row.faceid}"][:-(len(boundaryPts))]
        faceLandMarks = [(int(pt[0]), int(pt[1])) for pt in faceLandMarks]

        allDist = [calculate_distance(faceLandMarks[idx], averageFacialLandmarks[idx]) for idx in range(len(faceLandMarks))]
        distance = sum(allDist)
        imgfaces.keys()
        result[f"{row.imgid}-{row.faceid}"] = {"imgid": row.imgid, "faceid": row.faceid, "deviation": distance }

        output = output + img
    # Divide by numImages to get average
    output = output / faces.shape[0]


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

    
    facesresult = [ row for key, row in result.items()]
    def lmb_sort(x):
        key = f"{x['imgid']}-{x['faceid']}"
        return (result[key]['deviation'], result[key]['imgid'], result[key]['faceid'])
    facesresult = sorted(facesresult, key=lmb_sort)
    return {"groupkey": grpname, "faces": facesresult, "landmarks": averageFacialLandmarks}


def process_row(grpdata, faces_df, ofname, imgs_in_mem):
    grpname, grp = grpdata
    ids = list()
    if (isinstance(grpname, tuple)):
        grpname = "-".join(map(str, grpname))
    else:
        grpname = str(grpname)

    for row_index, row in grp.iterrows():
        ids.append(row_index)
    return process_transform(ids, grpname, faces_df, ofname, imgs_in_mem)


def process_dataframe(ofname, grouped_df, face_df, imgs_in_mem):
    rdir = os.path.join("results", ofname)
    ddir = os.path.join("debug", ofname)
    if not os.path.exists(rdir):
        os.makedirs(rdir)
    if not os.path.exists(ddir):
        os.makedirs(ddir)
    with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        no_downloaded = 0
        futures = {executor.submit(
            process_row, grp, face_df, ofname, imgs_in_mem): grp for grp in grouped_df}
        face_warp_hist = {}
        for future in concurrent.futures.as_completed(futures):
            result = futures[future]
            try:
                result = future.result()
                if (isinstance(result, dict)):
                    face_warp_hist[result["groupkey"]] = result["faces"]
            except Exception as e:
                print('generated an exception: %s' % (e))
            else:
                no_downloaded += 1
        print("finished files {0}".format(no_downloaded))
        with open("results/{0}.json".format(ofname), 'w') as outfile:
            json.dump(face_warp_hist, outfile)


faces_df = pd.read_json("faces.json").head(10)

# load images into memory
# load all raw files in mem:
imgs_in_mem = {}
for i, row in faces_df.iterrows():
    if row.imgid not in imgs_in_mem.keys():
        imgs_in_mem[row.imgid] = cv2.imread(os.path.join(
            "imgs", "{fid}.jpg".format(fid=row.imgid)))

omniart_df = pd.read_csv("omniart_v3_portrait.csv", encoding='utf8')

omniart_df.creation_year = pd.to_numeric(
    omniart_df.creation_year, downcast='integer')
omniart_df = omniart_df[omniart_df.id.isin(faces_df.imgid.unique())]

omnifaces_df = faces_df.set_index("imgid").join(omniart_df.set_index("id"))

# Warp images depending on group
omnifaces_df.sort_values(by="creation_year")
omnifaces_df["overall"] = "overall"
omnifaces_grouped = omnifaces_df.groupby(by=["overall"])
process_dataframe("overall", omnifaces_grouped, faces_df, imgs_in_mem)
print("overall")

omnifaces_df.sort_values(by="creation_year")
omnifaces_grouped = omnifaces_df.groupby(by=["gender"])
process_dataframe("overall-gender", omnifaces_grouped, faces_df, imgs_in_mem)
print("overall gender done")

omnifaces_df.sort_values(by="creation_year")
omnifaces_grouped = omnifaces_df.groupby(by=["age"])
process_dataframe("overall-age", omnifaces_grouped, faces_df, imgs_in_mem)
print("overall age done")

omnifaces_df.sort_values(by="creation_year")
omnifaces_grouped = omnifaces_df.groupby(by=["group"])
process_dataframe("overall-age", omnifaces_grouped, faces_df, imgs_in_mem)
print("overall group done")

# Warp images depending on group
omnifaces_df.sort_values(by="creation_year")
omnifaces_grouped = omnifaces_df.groupby(by=["creation_year"])
process_dataframe("yearly", omnifaces_grouped, faces_df, imgs_in_mem)
print("yearly done")

omnifaces_df.sort_values(by="creation_year")
omnifaces_grouped = omnifaces_df.groupby(by=["gender", "creation_year"])
process_dataframe("yearly-gender", omnifaces_grouped, faces_df, imgs_in_mem)
print("yearly gender done")

omnifaces_df.sort_values(by="creation_year")
omnifaces_grouped = omnifaces_df.groupby(by=["age", "creation_year"])
process_dataframe("yearly-age", omnifaces_grouped, faces_df, imgs_in_mem)
print("yearly age done")

omnifaces_df.sort_values(by="creation_year")
omnifaces_grouped = omnifaces_df.groupby(by=["group", "creation_year"])
process_dataframe("yearly-group", omnifaces_grouped, faces_df, imgs_in_mem)
print("yearly group done")

omnifaces_df["decade"] = omnifaces_df.creation_year.floordiv(10)
omnifaces_df.sort_values(by="decade")
omnifaces_grouped = omnifaces_df.groupby(by=["decade"])
process_dataframe("decade", omnifaces_grouped, faces_df, imgs_in_mem)
print("decade done")

omnifaces_grouped = omnifaces_df.groupby(by=["gender", "decade"])
process_dataframe("decade-gender", omnifaces_grouped, faces_df, imgs_in_mem)
print("decade gender done")

omnifaces_grouped = omnifaces_df.groupby(by=["age", "decade"])
process_dataframe("decade-age", omnifaces_grouped, faces_df, imgs_in_mem)
print("decade age done")

omnifaces_grouped = omnifaces_df.groupby(by=["group", "decade"])
process_dataframe("decade-group", omnifaces_grouped, faces_df, imgs_in_mem)
print("decade group done")

omnifaces_df["century"] = omnifaces_df.creation_year.floordiv(100)
omnifaces_df.sort_values(by="century")
omnifaces_grouped = omnifaces_df.groupby(by=["century"])
process_dataframe("century", omnifaces_grouped, faces_df, imgs_in_mem)
print("century done")

omnifaces_grouped = omnifaces_df.groupby(by=["gender", "century"])
process_dataframe("century-gender", omnifaces_grouped, faces_df, imgs_in_mem)
print("century gender done")

omnifaces_grouped = omnifaces_df.groupby(by=["age", "century"])
process_dataframe("century-age", omnifaces_grouped, faces_df, imgs_in_mem)
print("century age done")

omnifaces_grouped = omnifaces_df.groupby(by=["group", "century"])
process_dataframe("century-group", omnifaces_grouped, faces_df, imgs_in_mem)
print("century group done")
