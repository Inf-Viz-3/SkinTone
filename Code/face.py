
# Required modules
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

start=time.time()
min_YCrCb = np.array([0,133,77],np.uint8)
max_YCrCb = np.array([235,173,127],np.uint8)

# Get pointer to video frames from primary device
image = cv2.imread('D:/VU/Information_Visualization/untitled/data/baby1.png')
imageYCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
skinRegionYCrCb = cv2.inRange(imageYCrCb, min_YCrCb, max_YCrCb)



skinYCrCb = cv2.bitwise_and(image, image, mask = skinRegionYCrCb)


new_image = np.hstack([image, skinYCrCb])


cv2.imwrite('D:/VU/Information_Visualization/untitled/data/ycrcb.png', skinYCrCb)


def unique_count_app(a):
    colors, count = np.unique(a.reshape(-1,a.shape[-1]), axis=0, return_counts=True)
    return colors[count.argmax()]




# print(new_image[0])
#
# print(len(new_image))

print(np.shape(new_image))
color1=np.mean(new_image, axis=(0,1))

print(color1)
#
non_black=~(new_image == 0).all(2)
aa=new_image[non_black]
color2=np.mean(aa, axis=(0))
print(color2)

print(np.nanmean(np.where(new_image!=0,new_image,np.nan),axis=(0,1)))
# cv2.imwrite('D:/VU/Information_Visualization/untitled/data/5657.png', aa)
# print(aa)
# print(unique_count_app(aa))

# #
# print(non_black)
# print(new_image)
# print(unique_count_app(new_image))
#

print(time.time()-start)
