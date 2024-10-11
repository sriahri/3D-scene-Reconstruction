from osgeo import gdal 
import matplotlib.pyplot as plt 
import numpy as np

path = "/home/labmember/Desktop/project/EndoVis_dataset/dataset_1/keyframe_2/"
dataset = gdal.Open(path+"left_depth_map.tiff")

print(dataset.RasterCount)

# since there are 3 bands 
# we store in 3 different variables 
band1 = dataset.GetRasterBand(1) # Red channel 
band2 = dataset.GetRasterBand(2) # Green channel 
band3 = dataset.GetRasterBand(3) # Blue channel

b1 = band1.ReadAsArray() 
b2 = band2.ReadAsArray() 
b3 = band3.ReadAsArray() 

img = np.dstack((b1, b2, b3)) 
f = plt.figure() 
plt.imshow(img) 
plt.savefig('Tiff.png') 
plt.show() 

plt.imsave("depth_map.jpg")