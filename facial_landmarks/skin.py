import numpy as np
import cv2
from sklearn.cluster import KMeans
from collections import Counter
import pprint


def extractSkin(img):
    # Taking a copy of the image
    img = image.copy()
    # Converting from BGR Colours Space to HSV
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Defining HSV Threadholds
    lower_threshold = np.array([6, 34, 100], dtype=np.uint8)
    upper_threshold = np.array([40, 255, 255], dtype=np.uint8)

    # Single Channel mask,denoting presence of colours in the about threshold
    skinMask = cv2.inRange(img, lower_threshold, upper_threshold)

    # Cleaning up mask using Gaussian Filter
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)

    # Extracting skin from the threshold mask
    skin = cv2.bitwise_and(img, img, mask=skinMask)

    # Return the Skin image
    return cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)


def removeBlack(estimator_labels, estimator_cluster):

    # Check for black
    hasBlack = False

    # Get the total number of occurance for each color
    occurance_counter = Counter(estimator_labels)

    # Quick lambda function to compare to lists
    def compare(x, y): return Counter(x) == Counter(y)

    # Loop through the most common occuring color
    for x in occurance_counter.most_common(len(estimator_cluster)):

        # Quick List comprehension to convert each of RBG Numbers to int
        color = [int(i) for i in estimator_cluster[x[0]].tolist()]

        # Check if the color is [0,0,0] that if it is black
        if compare(color, [0, 0, 0]) == True:
            # delete the occurance
            del occurance_counter[x[0]]
            # remove the cluster
            hasBlack = True
            estimator_cluster = np.delete(estimator_cluster, x[0], 0)
            break

    return (occurance_counter, estimator_cluster, hasBlack)


def getColorInformation(estimator_labels, estimator_cluster, hasThresholding=False):

    # Variable to keep count of the occurance of each color predicted
    occurance_counter = None

    # Output list variable to return
    colorInformation = []

    # Check for Black
    hasBlack = False

    # If a mask has be applied, remove th black
    if hasThresholding == True:

        (occurance, cluster, black) = removeBlack(
            estimator_labels, estimator_cluster)
        occurance_counter = occurance
        estimator_cluster = cluster
        hasBlack = black

    else:
        occurance_counter = Counter(estimator_labels)

    # Get the total sum of all the predicted occurances
    totalOccurance = sum(occurance_counter.values())

    # Loop through all the predicted colors
    for x in occurance_counter.most_common(len(estimator_cluster)):

        index = (int(x[0]))

        # Quick fix for index out of bound when there is no threshold
        index = (index-1) if ((hasThresholding & hasBlack)
                              & (int(index) != 0)) else index

        # Get the color number into a list
        color = estimator_cluster[index].tolist()

        # Get the percentage of each color
        color_percentage = (x[1]/totalOccurance)

        # make the dictionay of the information
        colorInfo = {"cluster_index": index, "color": color,
                     "color_percentage": color_percentage}

        # Add the dictionary to the list
        colorInformation.append(colorInfo)

    return colorInformation


def extractDominantColor(image, number_of_colors=2, hasThresholding=False):

    # Quick Fix Increase cluster counter to neglect the black(Read Article)
    if hasThresholding == True:
        number_of_colors += 1

    # Taking Copy of the image
    img = image.copy()

    # Convert Image into RGB Colours Space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Reshape Image
    img = img.reshape((img.shape[0]*img.shape[1]), 3)

    # Initiate KMeans Object
    estimator = KMeans(n_clusters=number_of_colors, random_state=0)

    # Fit the image
    estimator.fit(img)

    # Get Colour Information
    colorInformation = getColorInformation(
        estimator.labels_, estimator.cluster_centers_, hasThresholding)


    return colorInformation


def plotColorBar(colorInformation):
    # Create a 500x100 black image
    color_bar = np.zeros((100, 500, 3), dtype="uint8")

    top_x = 0
    for x in colorInformation:
        bottom_x = top_x + (x["color_percentage"] * color_bar.shape[1])

        color = tuple(map(int, (x['color'])))

        cv2.rectangle(color_bar, (int(top_x), 0),
                      (int(bottom_x), color_bar.shape[0]), color, -1)
        top_x = bottom_x
    return color_bar


"""## Section Two.4.2 : Putting it All together: Pretty Print
The function makes print out the color information in a readable manner
"""


def prety_print_data(color_info):
    for x in color_info:
        print(pprint.pformat(x))
        print()

# # Get Image from URL. If you want to upload an image file and use that comment the below code and replace with  image=cv2.imread("FILE_NAME")
# image = imutils.url_to_image(
#     "https://uploads6.wikiart.org/a-soldier-1510(2).jpg")
#
# # Resize image to a width of 250
# image = imutils.resize(image, width=250)
#
# # Show image
# plt.subplot(3, 1, 1)
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.title("Original Image")
# # plt.show()
#
# # Apply Skin Mask
# skin = extractSkin(image)
#
# plt.subplot(3, 1, 2)
# plt.imshow(cv2.cvtColor(skin, cv2.COLOR_BGR2RGB))
# plt.title("Thresholded  Image")
# # plt.show()
#
# # Find the dominant color. Default is 1 , pass the parameter 'number_of_colors=N' where N is the specified number of colors
# dominantColors = extractDominantColor(skin, hasThresholding=True)
# Dominant=dominantColors[0].get('color')
# aa = []
# for i in range(len(Dominant)):
#     a=round(Dominant[i])
#     aa.append(a)
# print(aa)
# from csv import writer
#
# def append_list_as_row(file_name, list_of_elem):
#     # Open file in append mode
#     with open(file_name, 'a+', newline='') as write_obj:
#         # Create a writer object from csv module
#         csv_writer = writer(write_obj)
#         # Add contents of list as last row in the csv file
#         csv_writer.writerow(list_of_elem)
#
#
# csv_path='D:/VU/Information_Visualization/skin.csv'
# append_list_as_row(csv_path, aa)


#
# # Show in the dominant color information
# print("Color Information")
# prety_print_data(dominantColors)
#
# # Show in the dominant color as bar
# print("Color Bar")
# colour_bar = plotColorBar(dominantColors)
# plt.subplot(3, 1, 3)
# plt.axis("off")
# plt.imshow(colour_bar)
# plt.title("Color Bar")
#
# plt.tight_layout()
# plt.show()
#


import cv2
import os
import glob
from csv import writer
import pandas as pd

# data = pd.read_excel(r'D:/VU/Information_Visualization/untitled/met.xlsx')
# df = pd.DataFrame(data, columns= ['id', 'image_url'])



#The path to the folder with cropped images (faces)
img_dir = "D:/VU/Information_Visualization/Faces/"
data_path = os.path.join(img_dir,'*g')
files = glob.glob(data_path)
files.sort(key=lambda x: os.path.getmtime(x))
# image_labels1 = []
# for i in files:
#     img = cv2.imread(i)
#     head, tail = os.path.split(i)
#     image_labels1.append(tail[:-4])
#
#
# print(image_labels1.index("1c36ce10-30e9-4bc9-b2f6-36875c7e7e25"))
# print(image_labels1[9187:])
#

#Function to write to csv file
def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)

plt.rcParams.update({'figure.max_open_warning': 0})
# Loop for extracting the skin color, writing into csv file and saving plots
# for i in files[9187:]:
#     image = cv2.imread(i)
#     head, tail = os.path.split(i)
#     if len(image.shape) >= 3:
#         image = imutils.resize(image, width=250)
#         try:
#             skin = extractSkin(image)
#             dominantColors = extractDominantColor(skin, hasThresholding=True)
#             Dominant = dominantColors[0].get('color')
#             aa = []
#             for j in range(len(Dominant)):
#                 a = round(Dominant[j])
#                 aa.append(a)
#             print("Dominant color: {}. ".format(aa), 'Image: {}'.format(tail[:-4]))
#             new_row =[aa, tail[:-4]]
#             csv_path='D:/VU/Information_Visualization/skin.csv'
#             append_list_as_row(csv_path, new_row)
#             #plots
#             fig = plt.figure(figsize=(3, 6))
#             plt.subplot(3, 1, 1)
#             plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#             plt.title("Original Image")
#             plt.subplot(3, 1, 2)
#             plt.imshow(cv2.cvtColor(skin, cv2.COLOR_BGR2RGB))
#             plt.title("Thresholded  Image")
#             colour_bar = plotColorBar(dominantColors)
#             plt.subplot(3, 1, 3)
#             plt.axis("off")
#             plt.imshow(colour_bar)
#             plt.title("Color Bar")
#             plt.tight_layout()
#             fig.savefig("D:/VU/Information_Visualization/Plots/{}.png".format(tail[:-4]))
#
#         except Exception as e:
#             print(e)
#
#
##Here trying to check for the misclassified data
# data = pd.read_excel(r'D:/VU/Information_Visualization/untitled/skin_ex.xlsx')
# df = pd.DataFrame(data, columns= ['R', 'G', 'B', 'id'])
#
# misclassified=[]
# id=df['id']
# R=df['R']
# G=df['G']
# B=df['B']
#
# for i in range(len(id)):
#     if R[i] < 20:
#         if G[i] < 20:
#             if B[i] < 20:
#                 misclassified.append(id[i])
##Conclusion that if R, G, B all less than 20 most likely the second dominant color should be taken
## from Dominant = dominantColors[0].get('color') changed to Dominant = dominantColors[1].get('color')
# print(misclassified)
# print(len(misclassified))
# print(files)
# mis_data=[]
# for i in range(len(misclassified)):
#     file = "D:/VU/Information_Visualization/Faces\\" + misclassified[i] + ".jpg"
#     mis_data.append(file)
# print(mis_data)
#
##Doing the same for the misclassified data, but taking the second dominant color and writing into the other csv file
# ( and do not save plots)
# for i in mis_data:
#     image = cv2.imread(i)
#     head, tail = os.path.split(i)
#     if len(image.shape) >= 3:
#         image = imutils.resize(image, width=250)
#         try:
#             skin = extractSkin(image)
#             dominantColors = extractDominantColor(skin, hasThresholding=True)
#             Dominant = dominantColors[1].get('color')
#             aa = []
#             for j in range(len(Dominant)):
#                 a = round(Dominant[j])
#                 aa.append(a)
#             print("Dominant color: {}. ".format(aa), 'Image: {}'.format(tail[:-4]))
#             new_row =[aa, tail[:-4]]
#             csv_path='D:/VU/Information_Visualization/skin2.csv'
#             append_list_as_row(csv_path, new_row)
#         except Exception as e:
#             print(e)

##Here replacing in the first file the new data (changed misclassified images)
# data=pd.read_excel(r'D:/VU/Information_Visualization/untitled/skin_columns_1.xlsx')
# data_1=pd.read_excel(r'D:/VU/Information_Visualization/untitled/skin_columns_2.xlsx')
# df=pd.DataFrame(data)
# df_1=pd.DataFrame(data_1)
#
# SKIN=df['skin_color']
# SKIN_1=df_1['skin_color']
# ID=df['id']
# ID_1=df_1['id']
#
# for i in range(len(SKIN)):
#     for j in range(len(SKIN_1)):
#         if ID[i] == ID_1[j]:
#             print("First data set {}".format(ID[i]), "Second data set {}".format(ID_1[j]))
#             df.loc[ID == ID_1[j], "skin_color"] = SKIN_1[j]
#             df.to_excel("D:/VU/Information_Visualization/untitled/skin_columns_1.xlsx", index=False)
#
#



#Here the code to get hex colors of the skin for the slider
data=pd.read_excel(r'D:/VU/Information_Visualization/untitled/skin_columns_1.xlsx')
df=pd.DataFrame(data)
R=df['R']
G=df['G']
B=df['B']

def rgb_to_hex(rgb):
    return '#'+'%02X%02X%02X' % rgb
all_colors=[]
for i in range(len(R)):
    color=rgb_to_hex((R[i], G[i], B[i]))
    all_colors.append(color)

print(all_colors)
print(len(all_colors))

all_colors_1=list(dict.fromkeys(all_colors))
print(all_colors_1)
print(len(all_colors_1))

for i in all_colors_1:
    csv_path='D:/VU/Information_Visualization/color_skin_hex.csv'
    append_list_as_row(csv_path, i)
