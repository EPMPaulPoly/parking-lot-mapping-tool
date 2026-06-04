import os
import imageio.v2 as iio
import numpy as np


import rasterio
import os

import numpy as np

import warnings
warnings.filterwarnings("ignore")
from functions import convert_to_rgb, split_images


def processimage(input_folder,file_name):
    file_path = os.path.join(input_folder, file_name)
    with rasterio.open(file_path) as dataset:
        #print("CRS: ",dataset.crs)
        crs_in = dataset.crs
        #print("Dataset bound:",dataset.bounds)
        # Get image dimensions
        height, width = dataset.height, dataset.width
        #print("Height ", height)
        #print("Width ", width)
        # Generate grid of pixel indices (row, col)
        row_indices, col_indices = np.meshgrid(range(height), range(width), indexing="ij")

        # Transform pixel indices to lat/lon
        transform = dataset.transform
        lon, lat = rasterio.transform.xy(transform, row_indices, col_indices, offset="center")
        
        # Convert results to numpy arrays
        lat = np.array(lat).reshape(height, width)
        lon = np.array(lon).reshape(height, width)

    #print("Latitude shape:", lat.shape)
    #print("Longitude shape:", lon.shape)
    # Spliting the image into small images with size 512*512 to be ready for prediction
    height = width =512
    count = 0
    rgb_path = 'files/large_img.PNG'
    convert_to_rgb(file_path, rgb_path)
    img = iio.imread(rgb_path)
    count , img_shp1, img_shp2 = split_images(img, num = count)
    #print(img_shp1)
    #print(img_shp2)
    lons = lon[:img_shp1,:img_shp2]
    lats = lat[:img_shp1,:img_shp2]
    if img_shp1 > lons.shape[0]:
        pad_width = ((0, img_shp1 - lons.shape[0]), (0, 0))
        lons = np.pad(lons, pad_width, mode='edge')
        lats = np.pad(lats, pad_width, mode='edge')
    if img_shp2 > lons.shape[1]:
        pad_width = ((0, 0), (0, img_shp2 - lons.shape[1]))
        lons = np.pad(lons, pad_width, mode='edge')
        lats = np.pad(lats, pad_width, mode='edge')
    return lat,lon,lats,lons,crs_in,img_shp1,img_shp2,count