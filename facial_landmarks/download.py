import os
import pandas as pd
import concurrent.futures
import multiprocessing
import urllib.parse
import mimetypes
import requests
import cv2
import numpy as np

# use read.py to prepare portrait.csv
omniart_df = pd.read_csv("omniart_v3_portrait.csv", encoding = 'utf8')

def process_row(rowobj):
    row_data = rowobj[1]
    image_url = urllib.parse.urlparse(row_data['image_url'])
    filename, file_extension = os.path.splitext(image_url.geturl())
    image_resp = requests.get(image_url.geturl())
    nparr = np.fromstring(image_resp.content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cv2.imwrite("imgs/{0}.jpg".format(row_data.id), img)

with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
    no_downloaded = 0
    futures = { executor.submit(process_row, row,): row for row in omniart_df.iterrows() }
    for future in concurrent.futures.as_completed(futures):
        result = futures[future]
        try:
          result = future.result()
        except Exception as e:
            print('%r generated an exception: %s' % (result, e))
        else:
            no_downloaded+=1
    print("finished files {0}".format(no_downloaded))