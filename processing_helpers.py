from functions import convert_to_rgb, split_images, detect_polygons_inside, find_polygons, pixels_to_coordinates, poly_list_to_geojson
from post_processing import remove_buildings,get_road_data,remove_roads
import os
import geopandas as gpd
import rasterio
from shapely.geometry import Polygon, box
from pyproj import Transformer
from pathlib import Path

def basic_polygonize(lats,lons,seg_t,crs_in,name,out_path):
    # Converting large mask to polygons
    Polys, inner_polygons = find_polygons(seg_t)
    Polys_coord = pixels_to_coordinates(Polys, lons, lats)
    inner_coord = pixels_to_coordinates(inner_polygons, lons, lats)

    # Saving polygons in shape file format
    os.makedirs(out_path, exist_ok=True)
    output_path = out_path+'/' + name + '_original.geojson'
    poly_list_to_geojson(Polys_coord, inner_coord, output_path,crs_in)

def get_and_clean_buildings(input_path,name,output_path):
    parking_path = output_path+'/' + name + '_original.geojson'
    # parking_path = 'output_files/' + name + '_simp.shp'
    building_path = 'building_dataset/ref_queb_bat_montreal.geojson'
    build_state = gpd.read_file(building_path)
    file_name = input_path+'/'+name+'.tif'
    with rasterio.open(file_name) as src:
        bounds = src.bounds  # (minx, miny, maxx, maxy)
        bbox = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    bbox_gdf = gpd.GeoDataFrame({'geometry': [bbox]})
    bbox_gdf.set_geometry("geometry", inplace=True)
    bbox_gdf.set_crs('EPSG:3857', inplace=True)
    bbox_gdf.to_crs(build_state.crs, inplace=True)
    buildings = build_state[build_state.intersects(bbox_gdf.geometry.iloc[0], align=True)]
    buildings = buildings.iloc[0:0]
    output_file = output_path+'/' + name + '_build.shp'
    remove_buildings(parking_path, buildings, output_file)

def get_and_clean_roads(input_path,name,output_path):
    # Open TIFF file
    file_name = input_path+'/'+name+'.tif'
    with rasterio.open(file_name) as dataset:
        # Get bounding box in original CRS
        bbox = dataset.bounds  # BoundingBox(left, bottom, right, top)
        src_crs = dataset.crs  # Get CRS from the TIF file

    # Define target CRS (Lat/Lon)
    dst_crs = "EPSG:4326"

    # Initialize transformer
    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)

    # Convert bounding box coordinates
    min_lon, min_lat = transformer.transform(bbox.left, bbox.bottom)  # Bottom-left corner
    max_lon, max_lat = transformer.transform(bbox.right, bbox.top)  # Top-right corner

    # New bbox in Lat/Lon format
    converted_bbox = [min_lat, min_lon, max_lat, max_lon]
    print(converted_bbox)

    # Get road data
    get_road_data(converted_bbox)
    parking_path = output_path + '/' + name + '_build.shp'
    road_path = 'files/road_data.geojson'
    output_file = output_path + '/' + name + '_road.shp'
    remove_roads(parking_path, road_path, output_file)


def remove_holes(input_path,name,output_path):
    file_path = output_path + '/' + name + '_road.shp'
    gdf = gpd.read_file(file_path)
    gdf_exploded = gdf.explode(index_parts=False).reset_index(drop=True)
    gdf_exploded = gdf_exploded.to_crs(epsg=3857)
    gdf_all = gdf_exploded[gdf_exploded['geometry'].area > 300]
    gdf_all = gdf_all.to_crs(epsg=4326)
    gdf_all.to_file(output_path+'/' + name + '_all.geojson', driver='GeoJSON')

def cleanup_intermediate(output_path):
    folder = Path(output_path)
    for file in folder.iterdir():
        if file.is_file() and file.suffix.lower() in [".dbf", ".cpg",".prj",".shp",".shx"]:
            file.unlink()