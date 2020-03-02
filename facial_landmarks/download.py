import os
import pandas as pd
import concurrent.futures
import multiprocessing
import urllib.parse
import mimetypes
import requests

# use read.py to prepare portrait.csv
omniart_df = pd.read_csv("omniart_v3_portrait.csv", encoding = 'utf8')

def process_row(rowobj):
    row_data = rowobj[1]
    image_url = urllib.parse.urlparse(row_data['image_url'])
    filename, file_extension = os.path.splitext(image_url.geturl())
    image = requests.get(image_url.geturl())
    with open(f"imgs/{row_data.id}{file_extension}", 'wb') as fd:
        for chunk in image.iter_content(chunk_size=128):
            fd.write(chunk)

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