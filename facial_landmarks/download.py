import os
import pandas as pd
import concurrent.futures
import urllib.request

xlsx_data = pd.read_excel(os.path.join("..", "met.xlsx"))
df = pd.DataFrame(xlsx_data, columns= ['id', 'image_url'])

def process_row(rowobj):
    row_data = rowobj[1]
    if rowobj[0] % 1000 == 0:
        print("a thousand processed") 
    filename, file_extension = os.path.splitext(row_data['image_url'])
    urllib.request.urlretrieve(row_data['image_url'], "imgs/{0}{1}".format(row_data["id"], file_extension))   

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    no_downloaded = 0
    futures = { executor.submit(process_row, row,): row for row in df.iterrows() }
    for future in concurrent.futures.as_completed(futures):
        result = futures[future]
        try:
          result = future.result()
        except Exception as e:
            print('%r generated an exception: %s' % (result, e))
        else:
            no_downloaded+=1
    print("finished files {0}".format(no_downloaded))