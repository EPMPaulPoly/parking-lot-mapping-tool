import os
import gc
import cv2
import PIL
import imageio.v2 as iio
import torch
import random
import numpy as np
import multiprocessing as mp
from PIL import Image
import torchmetrics as tm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torchvision.transforms as tf
import torchvision.models.segmentation
import shapely
from PIL import ImageFilter
from numpy import asarray
from scipy.ndimage import measurements
from shapely import Polygon, LineString, MultiPolygon
from shapely.validation import explain_validity
import geopandas as gpd
from shapely.geometry import Polygon
import shutil
from shapely.ops import unary_union


def convert_to_rgba(tif_path, rgba_path):
    tif_image = Image.open(tif_path)
    rgba_image = tif_image.convert(mode='RGBA')
    rgba_image.save(rgba_path, "PNG", quality=100)


def convert_to_rgb(rgba_path, rgb_path):
    rgba_image = Image.open(rgba_path)
    rgb_image = rgba_image.convert(mode='RGB')
    rgb_image.save(rgb_path, "PNG", quality=100)


def split_images(img, im_size = 512, num = 0):
    """
    Split a large image into smaller tiles and save them.

    Args:
        img (np.ndarray): Input image as a NumPy array.
        im_size (int, optional): Size of the smaller tiles. Defaults to 512.
        num (int, optional): Initial numbering for the tiles. Defaults to 0.

    Returns:
        tuple: The updated tile number, and the final indices of rows and columns.
    """
    if os.path.exists('small_images'):
        shutil.rmtree('small_images')
    os.makedirs("small_images/Images", exist_ok=True)
    os.makedirs("small_images/Masks", exist_ok=True)
    
    img_h, img_w, img_c = img.shape
    first_ind, second_ind = 0, 0
    
    for i in range(0, img_h, im_size):
        first_ind = i + im_size
        for j in range(0, img_w, im_size):
            second_ind = j + im_size
            
            # Extract tile and pad if needed
            tile = img[i:i+im_size, j:j+im_size, :]
            h, w, _ = tile.shape
            
            if h < im_size or w < im_size:
                padded_tile = np.zeros((im_size, im_size, img_c), dtype=img.dtype)
                padded_tile[:h, :w, :] = tile
                tile = padded_tile
                
            num += 1
            Image.fromarray(tile).save('small_images/Images/' + str(num) + '.PNG', "PNG", quality=100)
            Image.fromarray(np.zeros((512, 512), dtype=np.uint8)).save('small_images/Masks/' + str(num) + '.PNG', "PNG", quality=100)
    return num , first_ind, second_ind



def detect_polygons_inside(bigger_polygon, polygons):
    polygons_inside = []
    indices_inside = []

    for i, polygon in enumerate(polygons):
        if polygon != bigger_polygon: 
            if polygon.is_valid and bigger_polygon.is_valid:
                if polygon.within(bigger_polygon): 
                    polygons_inside.append(polygon)
                    indices_inside.append(i)
                
    
    return polygons_inside, indices_inside


def find_polygons(seg_t):
    """
    Extract shapely polygons from a binary segmentation mask.

    Args:
        seg_t (np.ndarray): Binary segmentation mask.

    Returns:
        list: A list of Shapely Polygon objects.
    """
    # Filter small holes in Image format
    poly2 = (seg_t * 255).astype(np.uint8)
    image_try = Image.fromarray((poly2))
    image_try = image_try.filter(ImageFilter.ModeFilter(size=13))
    polygon_image = image_try.convert("L")
    polygon_im = asarray(polygon_image)


    # Find the separate polygons with value 1 and convert them to shapely polygons
    Polys = []
    a = cv2.findContours(polygon_im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    for i in range(len(a)):
        if a[i].shape[0] > 4:
            polygon = Polygon(np.squeeze(a[i]))
            if polygon.area > 1000:
                Polys.append(polygon)
    Polys = [polygon.buffer(0) if not polygon.is_valid else polygon for polygon in Polys]
    # Create a new list for inner polygons
    inner_polygons = []

    # Find polygons which are inside bigger polygons and merge them
    for i, polygon in enumerate(Polys):
        polygon_insides, indices_inside = detect_polygons_inside(polygon, Polys)
        if polygon_insides:
            # Add inner polygons to a new list (GeoDataFrame)
            inner_polygons.extend(polygon_insides)

            # Remove inner polygons from the main polygons list
            indices_inside.sort(reverse=True)
            for index in indices_inside:
                Polys.pop(index)

            # Merge inner polygons into the outer polygon
            Polys[i] = unary_union([Polys[i]] + polygon_insides)
                
    return Polys, inner_polygons


def pixels_to_coordinates(Polys, lons, lats):
    """
    Convert pixel-based polygons to coordinate-based polygons.

    Args:
        Polys (list): List of Shapely Polygon objects (pixel-based).
        lons (np.ndarray): 2D array of longitude values.
        lats (np.ndarray): 2D array of latitude values.

    Returns:
        list: A list of Shapely Polygon objects with geographic coordinates.
    """
    Polys_coord = []
    for n in range(len(Polys)):
        # Check if the geometry is a MultiPolygon
        if isinstance(Polys[n], MultiPolygon):
            for poly in Polys[n].geoms:  # Accessing the 'geoms' attribute to iterate through each individual polygon
                x, y = poly.exterior.coords.xy
                int_x = [int(x[i]) for i in range(len(x))]
                int_y = [int(y[i]) for i in range(len(y))]
                poly_lon = np.array([lons[int_y[i], int_x[i]] for i in range(len(x))])
                poly_lat = np.array([lats[int_y[i], int_x[i]] for i in range(len(y))])
                poly_list = list(zip(poly_lon, poly_lat))
                poly_coord = Polygon(poly_list)
                Polys_coord.append(poly_coord)
        else:
            # If it's a single Polygon, process it normally
            x, y = Polys[n].exterior.coords.xy
            int_x = [int(x[i]) for i in range(len(x))]
            int_y = [int(y[i]) for i in range(len(y))]
            poly_lon = np.array([lons[int_y[i], int_x[i]] for i in range(len(x))])
            poly_lat = np.array([lats[int_y[i], int_x[i]] for i in range(len(y))])
            poly_list = list(zip(poly_lon, poly_lat))
            poly_coord = Polygon(poly_list)
            Polys_coord.append(poly_coord)
    return Polys_coord


def poly_list_to_geojson(Polys_coord, inner_coord, output_path):
    """
    Save a list of polygons as a GeoJSON file.

    Args:
        Polys_coord (list): List of Shapely Polygon objects with geographic coordinates.
        output_path (str): Path to save the GeoJSON file.
    """
    # Create a GeoDataFrame
    data_inner = {'geometry': inner_coord}
    inner_gdf = gpd.GeoDataFrame(data_inner, geometry='geometry')
    data_outer = {'geometry': Polys_coord}
    outer_gdf = gpd.GeoDataFrame(data_outer, geometry='geometry')

    # Set the CRS (Coordinate Reference System)
    inner_gdf.set_crs('EPSG:3857', inplace=True)
    outer_gdf.set_crs('EPSG:3857', inplace=True)

    # Compute the union of all inner geometries
    inner_union = unary_union(inner_gdf['geometry'])

    # Subtract inner_union from each outer geometry
    outer_gdf['geometry'] = outer_gdf['geometry'].apply(lambda x: x.difference(inner_union))
    outer_gdf.to_file(output_path, driver='GeoJSON')