"""
This script downloads the cHL dataset (MAPS, Shaban et al.), prepares multi-channel TIFF images, copies cell mask files,
and extracts cell annotations.
"""

import os
import numpy as np
import pandas as pd
import tifffile as tiff
import skimage.io as skio
import shutil
import requests


def download_chl_maps_shaban_dataset(project_dir):
    """
    Downloads cHL dataset (MAPS, Shaban et al.) from Zenodo and saves it to the project dir.
    
    Paper: Pathologist-level Cell Type Annotation from Tissue Images through Machine Learning (Shaban et al., Nat. Comms.)
    Link: https://zenodo.org/records/10067010

    """
    url = "https://zenodo.org/records/10067010/files/cHL_CODEX.zip?"
    dataset_zip_path = os.path.join(project_dir, "dataset", "cHL_CODEX.zip")
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)
    if os.path.exists(dataset_zip_path):
        print("Dataset already exists. Skipping download.")
        return
    else:
        print("Downloading dataset from:", url)
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            os.makedirs(os.path.dirname(dataset_zip_path), exist_ok=True)
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024 * 1024  # 1MB
            progress = 0
            with open(dataset_zip_path, 'wb') as f:
                for data in response.iter_content(block_size):
                    f.write(data)
                    progress += len(data)
                    percent = progress * 100 // total_size if total_size else 0
                    print(f"\rDownloading: {percent}% ({progress // (1024*1024)}MB/{total_size // (1024*1024)}MB)", end="")
            print("\nDataset downloaded successfully.")
        else:
            print(f"Failed to download dataset. Status code: {response.status_code}")
            return
        os.system(f"unzip {dataset_zip_path} -d {os.path.join(project_dir, 'dataset')}")


def make_multi_channel_tiff(project_dir):
    marker_image_dir = os.path.join(project_dir, "dataset", "cHL_CODEX", "raw_image")
    os.makedirs(os.path.join(project_dir, "dataset", "multiplex_images"), exist_ok=True)
    if not os.path.exists(marker_image_dir):
        print("Raw image directory does not exist.")
        print("Downloading dataset...")
        download_chl_maps_shaban_dataset(project_dir)
    marker_images_list = os.listdir(marker_image_dir)
    marker_images_list = [img for img in marker_images_list if img.endswith('.tiff')]
    marker_images_list.sort()
    marker_images = []
    marker_info_dict = {"channel_id": [], "marker_name": []}
    for i, marker_image_name in enumerate(marker_images_list):
        marker_images.append(tiff.imread(os.path.join(marker_image_dir, marker_image_name)))
        marker_info_dict["channel_id"].append(i)
        marker_info_dict["marker_name"].append(marker_image_name.split('.')[0].upper())
    
    multi_channel_image = np.stack(marker_images, axis=0)  # Shape: (C, H, W)
    
    marker_info_df = pd.DataFrame(marker_info_dict)
    marker_info_df.to_csv(os.path.join(project_dir, "dataset", "marker_info.csv"), index=False)
    
    tiff.imwrite(os.path.join(project_dir, "dataset", "multiplex_images", "image_01.tiff"), multi_channel_image)
    print("Marker Image Type: ", marker_images[0].dtype, "Shape: ", marker_images[0].shape)
    
    image = skio.imread(os.path.join(project_dir, "dataset", "multiplex_images", "image_01.tiff"))
    if isinstance(image, np.ndarray) and image.ndim > 0:
        print("Marker Image Type: ", image[0].dtype, "Shape: ", image.shape)
    else:
        print("Error: The loaded image is not a multi-dimensional array.")


def copy_cell_mask_file(project_dir):
    """
    Copies the cell mask file to the project directory.
    """
    src_cell_mask_file_path = os.path.join(project_dir, "dataset", "cHL_CODEX", "segmentation", "cHL_CODEX_segmentation.tiff")
    des_cell_mask_file_path = os.path.join(project_dir, "dataset", "cell_masks", "image_01.tiff")
    if not os.path.exists(des_cell_mask_file_path):
        if not os.path.exists(src_cell_mask_file_path):
            print("Cell mask file does not exist in the source directory.")
            return
        os.makedirs(os.path.dirname(des_cell_mask_file_path), exist_ok=True)
        shutil.copy(src_cell_mask_file_path, des_cell_mask_file_path)
        print(f"Cell mask file copied to {des_cell_mask_file_path}")


def extract_cell_annotations(project_dir):
    """
    Extracts cell annotations from the dataset and saves them to a CSV file.
    """
    cell_annotations_file = os.path.join(project_dir, "dataset", "cHL_CODEX", "annotation_csv", "cHL_CODEX_annotation.csv")
    if not os.path.exists(cell_annotations_file):
        print("Cell annotations file does not exist.")
        return
    cell_type_names = ['B', 'CD4', 'CD8', 'DC', 'Endothelial', 'Epithelial', 'Lymphatic', 'M1', 'M2', 'Mast', 'Monocyte', 'NK', 'Neutrophil', 'Other', 'TReg', 'Tumor']
    cell_type_to_id_dict = {cell_type: i for i, cell_type in enumerate(cell_type_names)}
        
    annotations_df = pd.read_csv(cell_annotations_file)
    annotations_df = annotations_df[["cellLabel", "X_cent", "Y_cent", "cellSize", "cellType"]]
    annotations_df.columns = ["cell_id", "x_center", "y_center", "cell_size", "cell_type"]
    annotations_df = annotations_df[annotations_df["cell_type"]!="Seg Artifact"] # Filter out cells with segmentation artifacts
    annotations_df.loc[annotations_df["cell_type"]=="Cytotoxic CD8", "cell_type"] = "CD8" # Merging Cytotoxic CD8 to CD8
    annotations_df["cell_type_id"] = annotations_df["cell_type"].map(cell_type_to_id_dict)
    annotations_df["image_name"] = ["image_01" for _ in range(annotations_df.shape[0])] 
    annotations_df = annotations_df[["image_name", "cell_id", "x_center", "y_center", "cell_type", "cell_type_id"]] # Reorder columns

    annotations_df.to_csv(os.path.join(project_dir, "dataset", "cell_annotations.csv"), index=False)
    print(f"Total number of cells after filtering out segmentation artifacts and merging Cytotoxic CD8 to CD8: {len(annotations_df)}")
    print("Cell annotations extracted and saved to dataset/cell_annotations.csv")


def download_and_prepare_chl_maps_dataset(project_dir):
    """
    Downloads and prepares the cHL dataset by creating multi-channel TIFF images, copying cell mask files,
    and extracting cell annotations.
    """
    try:
        download_chl_maps_shaban_dataset(project_dir)
    except Exception as e:
        print(f"Error in download_dataset: {e}")
        return

    try:
        make_multi_channel_tiff(project_dir)
    except Exception as e:
        print(f"Error in make_multi_channel_tiff: {e}")
        return

    try:
        copy_cell_mask_file(project_dir)
    except Exception as e:
        print(f"Error in copy_cell_mask_file: {e}")
        return

    try:
        extract_cell_annotations(project_dir)
    except Exception as e:
        print(f"Error in extract_cell_annotations: {e}")
        return
    print("cHL dataset preparation completed successfully.")
