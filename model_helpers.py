from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from tqdm import tqdm
import torch
import numpy as np
from huggingface_hub import hf_hub_download
import numpy as np
from PIL import Image
from transformers import SegformerImageProcessor
from torch import nn
from torch.utils.data import DataLoader

from inference import SemanticSegmentationDataset, SegformerFinetuner, get_pred_segformer
import warnings
import shutil

warnings.filterwarnings("ignore")
# Function to process a batch of images


def import_model():
    repo_id = "UTEL-UIUC/SegFormer-large-parking"
    cached_file = hf_hub_download(repo_id=repo_id, filename="best_model.ckpt")
    shutil.copy(cached_file, "files/best_model.ckpt")

def load_model():

    feature_extractor = SegformerImageProcessor.from_pretrained(
        "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
    )

    feature_extractor.do_reduce_labels = False
    feature_extractor.size = 512

    path = "files/best_model.ckpt"

    model = SegformerFinetuner.load_from_checkpoint(
        path,
        id2label={0: "background", 1: "parking_lot"}  # adjust
    )

    model.eval()

    return model, feature_extractor

def build_dataloaders(img_shp1, img_shp2, feature_extractor):

    seg_t = np.zeros((img_shp1, img_shp2))

    n_r = img_shp1 // 512
    n_c = img_shp2 // 512

    test_dataset = SemanticSegmentationDataset(
        "small_images/",
        feature_extractor
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=1
    )

    return test_dataloader, n_c, n_r, seg_t, test_dataset

def process_batch(model,batch):
    images, masks = batch['pixel_values'], batch['labels']
    device = torch.device('cpu')
    images = images.to(device)
    masks = masks.to(device)
    with torch.no_grad():
        outputs = model.model(images, masks)
        logits = outputs[1]
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        predicted_mask = upsampled_logits.argmax(dim=1).cpu().numpy()
    return predicted_mask

def save_mask(test_dataset,img_number, mask):
    img_name = test_dataset.imgs[img_number].split('.')[0]
    Image.fromarray(mask.astype(np.uint8)).save('small_images/Masks/' + img_name + '.PNG', "PNG", quality=100)

def run_prediction(model:SegformerFinetuner,test_dataloader:DataLoader,count,n_c,n_r,seg_t,batch_size,img_shp1,img_shp2,test_dataset):

    # Move model to GPU and use DataParallel if multiple GPUs are available
    #if torch.cuda.device_count() > 1:
    #    segformer_finetuner.model = nn.DataParallel(segformer_finetuner.model)
    #segformer_finetuner.model.cuda()
    segformer_finetuner = model
    segformer_finetuner.model = segformer_finetuner.model.to('cpu')

    segformer_finetuner.model.eval() # Crucial for inference
    device = torch.device('cpu')

    # 2. Inference Loop (Sequential, but efficient with batching)
    # Assuming test_dataloader yields batches of images
    with torch.no_grad(): # Disables gradient calculation for speed/memory
        for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="Processing tiles of large image")):
            # Ensure inputs are on CPU
            if isinstance(batch, dict):
                batch = {k: v.to(device) for k, v in batch.items()}
            else:
                batch = batch.to(device)
                
            # Run inference
            masks = process_batch(segformer_finetuner,batch) # Your function that calls model(input)
            
            # Process and save
            for i in range(masks.shape[0]):
                img_number = batch_idx * batch_size + i
                save_mask( test_dataset,img_number, masks[i])

    # Assemble the final segmentation map
    for img_number in range(count):
        rr = img_number // n_c
        cc = img_number % n_c
        pred = Image.open('small_images/Masks/' + str(img_number+1) + '.PNG')
        seg_t[rr*512:(rr+1)*512, cc*512:(cc+1)*512] = pred


    seg_image = (seg_t*255)
    im = Image.fromarray(seg_image).convert('RGB')
    im.resize((int(img_shp2/10), int(img_shp1/10)))
    return seg_t

