from one_file_pipeline import one_file_pipeline,run_pipeline
from processing_helpers import cleanup_intermediate
from model_helpers import load_model
import os
from pathlib import Path
import shutil
from tqdm import tqdm

def main():
    model, feature_extractor = load_model()
    path = 'files/batch'
    outputdir = 'output_files/batch'
    for file_path in tqdm(os.listdir(path), desc="Processing large images"):
        folder = Path("./small_images")

        if folder.exists():
            shutil.rmtree(folder)

        folder.mkdir(parents=True, exist_ok=True)
        print(file_path)
        run_pipeline(path,file_path,outputdir,model,feature_extractor)
        cleanup_intermediate(outputdir)
if __name__ =='__main__':
    main()