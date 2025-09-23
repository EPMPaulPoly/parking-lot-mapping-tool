# Parking Lot Segmentation Tool  

## Overview  
This repository provides a **parking lot segmentation tool** that detects and delineates parking lots from large satellite images.  
The tool uses a SegFormer-large model trained on the dataset introduced in our [WACV 2025 paper](https://ieeexplore.ieee.org/abstract/document/10943617) *“A Pipeline and NIR-Enhanced Dataset for Parking Lot Segmentation.”*  

The pipeline includes both **deep learning segmentation** and **post-processing**, producing a **GeoJSON file** containing polygons for detected parking lots.  

- **Input**: large satellite image (e.g., GeoTIFF)  
- **Output**: GeoJSON file of parking lot polygons  

The pretrained weights are hosted on Hugging Face: [UTEL-UIUC/SegFormer-large-parking](https://huggingface.co/UTEL-UIUC/SegFormer-large-parking).  

---

## Getting Started  

### 1. Clone the repository  
```bash
git clone https://github.com/UTEL-UIUC/parking-lot-segmentation-tool.git
cd parking-lot-segmentation-tool
```

### 2. Install dependencies

Install the required packages:
```bash
pip install -r requirements.txt
```
### 3. Run the tool

The main entry point is a Jupyter Notebook:

```bash
jupyter notebook main.ipynb
```

Open the notebook and follow the steps to:

- Prepare your satellite image for your area of interest
- Run the cells in the notebook to:
  1. Get the predictions from the model
  2. Perform the post-processing
  3. Export the result as a GeoJSON file

---
## Citation

If you use this tool or the pretrained model, please cite:

```bibtex
@inproceedings{qiam2025pipeline,
  title={A Pipeline and NIR-Enhanced Dataset for Parking Lot Segmentation},
  author={Qiam, Shirin and Devunuri, Saipraneeth and Lehe, Lewis J},
  booktitle={2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
  pages={1227--1236},
  year={2025},
  organization={IEEE}
}
```

