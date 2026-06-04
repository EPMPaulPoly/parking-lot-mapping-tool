
from image_helpers import processimage
from model_helpers import build_dataloaders,run_prediction,load_model
import os
from pathlib import Path
from processing_helpers import basic_polygonize,get_and_clean_buildings,get_and_clean_roads,remove_holes
def run_pipeline(input_path,filename,output_path,model_in,feature_extractor):

    name = Path(filename).stem
    # pre process image
    batch_size =1
    lat,lon,lats,lons,crs_in,img_shp1,img_shp2,count= processimage(input_path,filename)
    # create model
    dataloader, n_c, n_r, seg_t, test_dataset= build_dataloaders(img_shp1,img_shp2,feature_extractor)
    # run prediction
    seg_t = run_prediction(model_in,dataloader,count,n_c,n_r,seg_t,batch_size,img_shp1,img_shp2,test_dataset)
    # create_polygon
    basic_polygonize(lats,lons,seg_t,crs_in,name,output_path)
    get_and_clean_buildings(input_path,name,output_path)
    get_and_clean_roads(input_path,name,output_path)
    remove_holes(input_path,name,output_path)


def one_file_pipeline(input_path,filename,output_path):
    model, feature_extractor = load_model()
    run_pipeline(input_path,filename,output_path,model,feature_extractor)

if __name__=="__main__":
    input_path = 'files/batch'
    file = '268-5031_aout23_rgb_25cm.tif'
    output_path = 'output_files/batch'
    #one_file_pipeline(input_path,file,output_path)
    name = Path(file).stem
    get_and_clean_buildings(input_path,name,output_path)
    get_and_clean_roads(input_path,name,output_path)
    remove_holes(input_path,name,output_path)
    
