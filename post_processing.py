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
import shapely
from PIL import ImageFilter
from numpy import asarray
from scipy.ndimage import measurements
from shapely import Polygon, LineString, MultiPolygon
from shapely.validation import explain_validity
import geopandas as gpd
from shapely.geometry import Polygon
import subprocess
from shapely.ops import unary_union
import requests
import zipfile
import pandas as pd
import requests
import json
from geojson import Feature, FeatureCollection, LineString

    
    
def remove_buildings(parking_path, buildings, output_path):
    # Load the shapefiles
    parking_lots = gpd.read_file(parking_path)
    parking_lots = parking_lots.set_crs('EPSG:3857')

    #Ensure both GeoDataFrames have the same coordinate reference system (CRS)
    if parking_lots.crs != buildings.crs:
        buildings = buildings.to_crs(parking_lots.crs)
          
    parking_lots = parking_lots[parking_lots['geometry'].notna()]
    buildings = buildings[buildings['geometry'].notna()]

    # Clean the geometries to avoid self-intersection issues
    parking_lots['geometry'] = parking_lots['geometry'].buffer(0)
    buildings['geometry'] = buildings['geometry'].buffer(0)

    # Combine all building polygons into a single geometry
    buildings_union = unary_union(buildings['geometry'])

    # Perform the difference operation
    corrected_parking_lots = parking_lots.copy()
    corrected_parking_lots['geometry'] = corrected_parking_lots['geometry'].apply(lambda x: x.difference(buildings_union))

    # Save the corrected parking lot polygons to a new shapefile
    corrected_parking_lots.to_file(output_path)
    
    
def remove_roads(parking_path, road_path, output_path):
    # Load the shapefiles
    parking_lots = gpd.read_file(parking_path)
    roads = gpd.read_file(road_path)

    # Ensure both GeoDataFrames have the same coordinate reference system (CRS)
    if parking_lots.crs != roads.crs:
        roads = roads.to_crs(parking_lots.crs)

    parking_lots = parking_lots[parking_lots['geometry'].notna()]
    roads = roads[roads['geometry'].notna()]

    # Clean the geometries to avoid self-intersection issues
    parking_lots['geometry'] = parking_lots['geometry'].buffer(0)
    roads['geometry'] = roads['geometry'].buffer(0)

    # Combine all building polygons into a single geometry
    roads_union = unary_union(roads['geometry'])

    # Perform the difference operation
    corrected_parking_lots = parking_lots.copy()
    corrected_parking_lots['geometry'] = corrected_parking_lots['geometry'].apply(lambda x: x.difference(roads_union))

    # Save the corrected parking lot polygons to a new shapefile
    corrected_parking_lots.to_file(output_path)
    
    
def get_building_data(state_name, dataset_folder, csv_file):
    state_name_full = state_name + ".geojson.zip"
    # Check if a file with the state name exists in the folder
    zip_file_path = os.path.join(dataset_folder, state_name_full)
    if os.path.exists(zip_file_path):
        print(f"Building data for {state_name} already exists.")
        return

    # Load the CSV to get the download URL
    building_links = pd.read_csv(csv_file)
    
    # Find the URL for the given state
    row = building_links[building_links['State'].str.lower() == state_name.lower()]
    if row.empty:
        print(f"No URL found for state: {state_name}")
        return
    
    url = row.iloc[0]['URL']
    
    # Download the file
    response = requests.get(url, stream=True)
    with open(zip_file_path, 'wb') as file:
        for chunk in response.iter_content(chunk_size=1024):
            file.write(chunk)
    print(f"Downloaded {zip_file_path}.")
    
    # Unzip the file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_folder)
        
    print(f"Extracted {zip_file_path}.")
        
    
        
        
def osm_to_geojson(osm_data):
    features = []
    
    # Create a dictionary to store node coordinates
    node_dict = {node['id']: (node['lon'], node['lat']) for node in osm_data.get("elements", []) if node["type"] == "node"}

    # Convert ways to LineStrings
    for element in osm_data.get("elements", []):
        if element["type"] == "way":
            coords = [node_dict[node_id] for node_id in element["nodes"] if node_id in node_dict]
            if coords:  # Ensure we have valid coordinates
                feature = Feature(
                    geometry=LineString(coords),
                    properties=element.get("tags", {})
                )
                features.append(feature)

    return FeatureCollection(features)


def get_road_data(bbox):
    # Overpass API query
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = f"""
    [out:json][timeout:25];
    (
    way["highway"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
    relation["highway"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
    );
    out body;
    >;
    out skel qt;
    """

    response = requests.get(overpass_url, params={"data": overpass_query})

    if response.status_code == 200:
        osm_data = response.json()

    else:
        print("Error fetching OSM data:", response.status_code, response.text)

    geojson_data = osm_to_geojson(osm_data)

    # Save the linestring data to file
    with open("files/road_data_line.geojson", "w", encoding="utf-8") as f:
        json.dump(geojson_data, f, indent=2)
        
        
    # Create a buffer around the lines based on number of lanes
    gdf = gpd.read_file('files/road_data_line.geojson')
    exclude_categories = ['service', 'footway', 'path', 'performed', 'construction', 'steps', 'track']
    if gdf.empty:
        gdf.to_file('files/road_data.geojson', driver='GeoJSON')
    else:
        gdf = gdf[~gdf['highway'].isin(exclude_categories)]
        if gdf.empty:
            gdf.to_file('files/road_data.geojson', driver='GeoJSON')
        else:
            if 'lanes' not in gdf: gdf['lanes'] = 1
            gdf['lanes'] = gdf['lanes'].fillna(1)
            gdf['width_buffer'] = gdf.apply(lambda row: int(row['lanes']) * 1 if row['highway'] == 'cycleway' else int(row['lanes']) * 3, axis=1)
            gdf_p = gdf.to_crs(epsg=3857)
            gdf_p['geometry'] = gdf_p.apply(lambda row: row['geometry'].buffer(row['width_buffer'], cap_style='flat'), axis=1)
    
    # Save the road data
    gdf_p.to_file('files/road_data.geojson', driver='GeoJSON')
    print("GeoJSON saved to files/road_data.geojson")