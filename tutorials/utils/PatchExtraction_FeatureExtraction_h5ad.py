#!/usr/bin/env python
"""
Patch Extraction and Feature Extraction Utilities

This file provides utilities for extracting patches from multiplex images and 
extracting features from those patches using pre-trained models.

CSV Format Requirements:
    The marker metadata CSV file should contain the following columns:
    - image_channel (int): Channel number in the multiplex image (1-indexed)
    - marker_name (str): Name of the marker/protein
    - marker_id (int): Unique index for the marker (used in feature extraction)
    - marker_mean (float): Mean value for marker normalization (used in feature extraction)
    - marker_std (float): Standard deviation for marker normalization (used in feature extraction)
    
    Example CSV format:
    image_channel,marker_name,marker_id,marker_mean,marker_std
    1,DAPI,0,0.1234,0.0567
    2,CD3,1,0.0891,0.0432
    3,CD8,2,0.0654,0.0321
    4,PD1,3,0.0432,0.0198

Classes:
    PatchExtraction:
        Handles extraction of patches from multiplex images with tissue segmentation masks.
        Saves patches as HDF5 files containing individual marker datasets.
    
    FeatureExtraction:
        Handles feature extraction from patches using pre-trained models.
        Loads patches and extracts features, saving them as numpy arrays.

    H5ADBuilder:
        Handles building AnnData (h5ad) objects from numpy embedding files.
        Loads numpy embedding files, extracts metadata from filenames,
        merges with sample-level metadata, and saves as h5ad objects for downstream analysis.
        
"""

import os
import sys
import h5py
import csv
import numpy as np
import pandas as pd
import skimage.io as skio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# from torchvision.transforms import v2
import scanpy as sc
from glob import glob
from kronos import create_model_from_pretrained


def load_model(self):
        """
        Loads a pre-trained model using the KRONOS library.

        The model is loaded based on the configuration parameters provided in the `config` dictionary.
        This includes the checkpoint path, Hugging Face authentication token, cache directory, and model-specific settings.

        Returns:
            model: The pre-trained model loaded from the specified checkpoint.
            precision: The precision of the model (e.g., float32, float16).
            embedding_dim: The dimensionality of the model's output embeddings.
        """
        # Use the KRONOS library to create a model from a pre-trained checkpoint
        model, precision, embedding_dim = create_model_from_pretrained(
            checkpoint_path=self.config["checkpoint_path"],  # Path to the model checkpoint
            cfg_path=None,  # Configuration path (not used here)
            hf_auth_token=self.config["hf_auth_token"],  # Hugging Face authentication token
            cache_dir=self.config["cache_dir"],  # Directory to cache the model
            cfg={
                "model_type": self.config["model_type"],  # Type of model (e.g., ViT-S16)
                "token_overlap": self.config["token_overlap"]  # Whether to use token overlap
            },
        )
        return model, precision, embedding_dim


class PatchExtraction:
    """
    A class for extracting patches from multiplex images.
    
    This class handles the extraction of patches from multiplex images using a sliding
    window approach across the entire image. Each patch is saved as an HDF5 file 
    containing individual marker datasets.
    
    Attributes:
        config (dict): Configuration dictionary containing extraction parameters.
        image_dir (str): Directory containing the input images.
        output_dir (str): Directory to save the extracted patches.
        marker_list (list): List of marker information dictionaries.
        patch_size (int): Size of patches to extract (default: 256).
        stride (int): Stride for patch extraction (default: 256).
        file_ext (str): File extension of input images (default: '.ome.tiff').
        device (torch.device): Device to use for processing (GPU/CPU).
    """
    
    def __init__(self, config):
        """
        Initializes the PatchExtraction class.
        
        Args:
            config (dict): Configuration dictionary containing:
                - image_dir (str): Directory containing input images
                - output_dir (str): Directory to save patches
                - marker_csv_path (str): Path to marker metadata CSV
                - patch_size (int, optional): Size of patches (default: 256)
                - stride (int, optional): Stride for extraction (default: 256)
                - file_ext (str, optional): File extension (default: '.ome.tiff')
        """
        self.config = config
        self.image_dir = config["image_dir"]
        self.output_dir = config["output_dir"]
        self.patch_size = config.get("patch_size", 256)
        self.stride = config.get("stride", 256)
        self.file_ext = config.get("file_ext", ".ome.tiff")
        
        # Load marker information
        self.marker_list = self._read_marker_csv(config["marker_csv_path"])
        
        # Set up device (let PyTorch automatically choose)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            print("No GPU available, using CPU instead")
    
    def _read_marker_csv(self, csv_path):
        """
        Read marker metadata from CSV file.
        
        Args:
            csv_path (str): Path to the marker CSV file.
            
        Returns:
            list: List of dictionaries containing marker information.
        """
        marker_list = []
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    marker_list.append({
                        'image_channel': int(row['image_channel']),
                        'marker_name': row['marker_name']
                    })
            return marker_list
        except Exception as e:
            print(f"Error reading marker CSV: {str(e)}")
            sys.exit(1)
    
    def extract_patches_from_image(self, img_name):
        """
        Extract patches from a single image across the entire core.
        
        Args:
            img_name (str): Name of the image file to process.
            
        Returns:
            int: Number of patches extracted from the image.
        """
        # Create output subdirectory
        out_subdir = os.path.join(self.output_dir, "%d_%d" % (self.patch_size, self.stride))
        os.makedirs(out_subdir, exist_ok=True)
        
        # Skip if not a valid image file
        if not img_name.endswith(self.file_ext):
            print(f"Skipping {img_name} as it is not a {self.file_ext} file")
            return 0
        
        # Skip if output files already exist
        base_name = os.path.splitext(img_name)[0]
        existing_files = [f for f in os.listdir(out_subdir) if f.startswith(f"{base_name}_")]
        if existing_files:
            print(f"Skipping {img_name} as patches already exist")
            return len(existing_files)
        
        try:
            # Read the image
            img = skio.imread(os.path.join(self.image_dir, img_name))
            print(f"Processing {img_name} with shape {img.shape}")
            
            # Convert image to torch tensor and move to device
            img_tensor = torch.from_numpy(img).to(self.device)
            
            # Extract patches from the entire image
            count = 0
            for i in range(0, img.shape[1]-self.patch_size+1, self.stride):
                for j in range(0, img.shape[2]-self.patch_size+1, self.stride):
                    # Create h5 file for the patch
                    patch_file = os.path.join(out_subdir, "%s_%06d_%06d.h5" % (base_name, i, j))
                    with h5py.File(patch_file, "w") as h5f:
                        # Save each marker channel as a separate dataset
                        for marker_info in self.marker_list:
                            # Get the channel index (1-indexed in the CSV, so subtract 1)
                            channel_idx = marker_info['image_channel'] - 1
                            
                            # Make sure the channel index is valid
                            if channel_idx < 0 or channel_idx >= img.shape[0]:
                                print(f"Skipping invalid channel {channel_idx} for marker {marker_info['marker_name']}")
                                continue
                            
                            try:
                                # Extract patch from the 3D image (channels, height, width)
                                patch_tensor = img_tensor[channel_idx, i:i+self.patch_size, j:j+self.patch_size]
                                # Move back to CPU for saving
                                patch = patch_tensor.cpu().numpy()
                                h5f.create_dataset(marker_info['marker_name'], data=patch)
                            except Exception as e:
                                print(f"Error extracting patch at ({i},{j}) for marker {marker_info['marker_name']}: {str(e)}")
                                continue
                    count += 1
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"Extracted {count} patches from {img_name}")
            return count
            
        except Exception as e:
            print(f"Error processing {img_name}: {str(e)}")
            # Clear GPU memory on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return 0
    
    def extract_all_patches(self, file_list=None):
        """
        Extract patches from all images in the directory or from a specified file list.
        
        Args:
            file_list (list, optional): List of image filenames to process. 
                                      If None, processes all files in image_dir.
        
        Returns:
            dict: Dictionary with image names as keys and patch counts as values.
        """
        # If file_list is provided, use it; otherwise process all files in the directory
        if file_list is None:
            file_list = [f for f in os.listdir(self.image_dir) if f.endswith(self.file_ext)]
        
        results = {}
        total_files = len(file_list)
        
        for idx, img_name in enumerate(file_list):
            print(f"Processing file {idx+1}/{total_files}: {img_name}")
            patch_count = self.extract_patches_from_image(img_name)
            results[img_name] = patch_count
        
        print("Patch extraction completed!")
        return results


class MultiplexDataset(Dataset):
    """
    Custom Dataset class for loading multiplex image patches.
    
    Attributes:
        data_dir (str): Directory containing the image patches.
        patch_list (list): List of patch filenames.
        marker_info (pd.DataFrame): DataFrame containing marker information.
        nuclear_stain (str): Name of the nuclear stain marker.
        marker_list (list): List of marker names.
        transform (callable, optional): Optional transform to be applied on a sample.
        max_value (float): Maximum value for normalization.
    """
    
    def __init__(self, data_dir, patch_list, marker_info, nuclear_stain, marker_list, max_value=65535, transform=None):
        """
        Args:
            data_dir (str): Directory containing the image patches.
            patch_list (list): List of patch filenames.
            marker_info (str): Path to the CSV file containing marker information.
            nuclear_stain (str): Name of the nuclear stain marker.
            marker_list (list): List of marker names.
            max_value (float, optional): Maximum value for normalization. Default is 65535.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.patch_list = patch_list
        
        self.marker_info = pd.read_csv(marker_info)
        self.marker_info = self.marker_info.set_index('marker_name')
        
        self.nuclear_stain = nuclear_stain
        self.marker_list = marker_list
        
        self.max_value = max_value
        self.transform = transform
    
    def __len__(self):
        """Returns the total number of patches."""
        return len(self.patch_list)
    
    def __getitem__(self, index):
        """
        Retrieves a patch and its corresponding marker IDs.
        
        Args:
            index (int): Index of the patch to retrieve.
            
        Returns:
            tuple: (patch, marker_ids, patch_name, patch_markers)
        """
        patch_name = self.patch_list[index]
        patch, patch_marker_ids, patch_markers = self.get_patch(os.path.join(self.data_dir, patch_name))
        if self.transform is not None:
            patch = self.transform(patch)
        marker_ids = torch.tensor(patch_marker_ids.tolist())
        return patch, marker_ids, patch_name, patch_markers
    
    def get_patch(self, patch_path):
        """
        Loads a patch and normalizes it using marker statistics.
        
        Args:
            patch_path (str): Path to the patch file.
            
        Returns:
            tuple: (patch, patch_marker_ids, patch_markers)
        """
        patch_markers = [self.nuclear_stain] + self.marker_list
        patch_marker_ids = [self.marker_info.loc[marker_name, "marker_id"] for marker_name in patch_markers]
        marker_patch = []
        
        with h5py.File(patch_path, 'r') as f:
            # Convert all keys in the h5 file to lowercase for case-insensitive matching
            h5_keys_lower = {k.lower(): k for k in f.keys()}
            
            for marker_name in patch_markers:
                marker_mean = self.marker_info.loc[marker_name, "marker_mean"]
                marker_std = self.marker_info.loc[marker_name, "marker_std"]
                
                # Look for the marker in the lowercase keys dictionary
                marker_key_lower = marker_name.lower()
                if marker_key_lower in h5_keys_lower:
                    # Use the original case key from the file
                    original_key = h5_keys_lower[marker_key_lower]
                    patch = f[original_key][:]/self.max_value
                else:
                    raise KeyError(f"Marker '{marker_name}' not found in file (case-insensitive search failed)")
                
                patch = (patch-marker_mean)/marker_std
                marker_patch.append(torch.tensor(patch))
                
            patch = torch.stack(marker_patch, dim=0)
            patch_marker_ids = np.uint16(patch_marker_ids)
        
        return patch, patch_marker_ids, patch_markers


class FeatureExtraction:
    """
    A class for extracting features from multiplex image patches using pre-trained models.
    
    This class handles loading patches, applying pre-trained models to extract features,
    and saving the resulting features as numpy arrays for downstream analysis.
    
    Attributes:
        config (dict): Configuration dictionary containing extraction parameters.
        dataset_dir (str): Directory containing the patch files.
        feature_dir (str): Directory to save extracted features.
        model (torch.nn.Module): Pre-trained model for feature extraction.
        device (torch.device): Device to use for processing (GPU/CPU).
        marker_info (str): Path to marker metadata CSV file.
        nuclear_stain (str): Name of the nuclear stain marker.
        marker_list (list): List of marker names to process.
    """
    
    def __init__(self, config):
        """
        Initializes the FeatureExtraction class.
        
        Args:
            config (dict): Configuration dictionary containing:
                - dataset_dir (str): Directory containing patch files
                - feature_dir (str): Directory to save features
                - checkpoint_path (str): Path to model checkpoint
                - hf_auth_token (str): Hugging Face authentication token
                - cache_dir (str): Directory to cache the model
                - model_type (str): Type of model (e.g., ViT-S16)
                - token_overlap (bool): Whether to use token overlap
                - marker_info (str): Path to marker metadata CSV
                - nuclear_stain (str, optional): Nuclear stain marker name (default: 'DAPI')
                - max_value (float, optional): Maximum value for normalization (default: 65535.0)
                - batch_size (int, optional): Batch size for processing (default: 3)
                - num_workers (int, optional): Number of workers for data loading (default: 4)
        """
        self.config = config
        self.dataset_dir = config["dataset_dir"]
        self.feature_dir = config["feature_dir"]
        self.marker_info = config["marker_info"]
        self.nuclear_stain = config.get("nuclear_stain", "DAPI")
        self.max_value = config.get("max_value", 65535.0)
        self.batch_size = config.get("batch_size", 3)
        self.num_workers = config.get("num_workers", 4)
        
        # Set up device (let PyTorch automatically choose)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            print("No GPU available, using CPU instead")
        
        # Load marker list
        marker_df = pd.read_csv(self.marker_info)
        self.marker_list = marker_df["marker_name"].tolist()
        if self.nuclear_stain in self.marker_list:
            self.marker_list.remove(self.nuclear_stain)
        
        # Load model
        self.model = self._get_kronos_model()
    
    def _get_kronos_model(self):
        """
        Loads the Kronos model from checkpoint using the new Kronos library.
        
        Returns:
            torch.nn.Module: Loaded model.
        """
        print("Loading Kronos model...")
        model, precision, embedding_dim = create_model_from_pretrained(
            checkpoint_path=self.config["checkpoint_path"],  # Path to the model checkpoint
            cfg_path=None,  # Configuration path (not used here)
            hf_auth_token=self.config["hf_auth_token"],  # Hugging Face authentication token
            cache_dir=self.config["cache_dir"],  # Directory to cache the model
            cfg={
                "model_type": self.config["model_type"],  # Type of model (e.g., ViT-S16)
                "token_overlap": self.config["token_overlap"]  # Whether to use token overlap
            },
        )
        
        model.to(self.device)
        model.eval()
        print(f"Model loaded successfully with precision: {precision}, embedding dim: {embedding_dim}")
        return model
    
    def extract_features_from_patches(self, patch_list=None, token_features=True):
        """
        Extract features from patches using the pre-trained model.
        
        Args:
            patch_list (list, optional): List of patch filenames to process.
                                       If None, processes all .h5 files in dataset_dir.
            token_features (bool, optional): Whether to save token features. Default is True.
        
        Returns:
            int: Number of patches processed.
        """
        # Get patch list
        if patch_list is None:
            patch_list = os.listdir(self.dataset_dir)
            patch_list = [patch for patch in patch_list if patch.endswith(".h5")]
        
        print(f"Found {len(patch_list)} patches to process")
        
        if len(patch_list) == 0:
            print(f"No patches found in {self.dataset_dir}. Exiting.")
            return 0
        
        # Create dataset and dataloader
        dataset = MultiplexDataset(
            self.dataset_dir, 
            patch_list, 
            self.marker_info, 
            self.nuclear_stain, 
            self.marker_list, 
            self.max_value, 
            transform=None
        )
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers
        )
        
        # Create feature directories
        os.makedirs(self.feature_dir, exist_ok=True)
        os.makedirs(os.path.join(self.feature_dir, 'norm_clstoken'), exist_ok=True)
        if token_features:
            os.makedirs(os.path.join(self.feature_dir, 'norm_patchtokens'), exist_ok=True)
        
        # Process batches
        batch_count = len(dataloader)
        processed_patches = 0
        
        for i, (img_batch, marker_indices, image_batch, patch_markers_batch) in enumerate(dataloader):
            print(f"Processing batch {i + 1}/{batch_count}")
            
            img_batch = img_batch.to(self.device)
            img_batch = img_batch.to(torch.float)
            marker_indices = marker_indices.to(self.device)
            
            with torch.no_grad():
                # Use the Kronos model to extract features
                # The output format may vary, so we need to handle different cases
                output = self.model(img_batch, marker_ids=marker_indices)
                
                # Handle different output formats from Kronos model
                if isinstance(output, dict):
                    # If output is a dictionary (like the old format)
                    norm_clstoken = output.get('x_norm_clstoken', output.get('cls_token', None))
                    norm_patchtokens = output.get('x_norm_patchtokens', output.get('patch_tokens', None))
                elif isinstance(output, tuple) and len(output) >= 2:
                    # If output is a tuple (cls_features, marker_features, ...)
                    norm_clstoken = output[0]  # cls features
                    norm_patchtokens = output[1]  # marker features
                else:
                    # If output is a single tensor, assume it's cls token
                    norm_clstoken = output
                    norm_patchtokens = None
                
                # Convert to numpy
                if norm_clstoken is not None:
                    norm_clstoken = norm_clstoken.cpu().numpy()
                if norm_patchtokens is not None:
                    norm_patchtokens = norm_patchtokens.cpu().numpy()
                    token_per_marker = norm_patchtokens.shape[1] // len(patch_markers_batch)
                    feat_dim = norm_clstoken.shape[-1] if norm_clstoken is not None else norm_patchtokens.shape[-1]
                
                for j, file_name in enumerate(image_batch):
                    # Save class token features if available
                    if norm_clstoken is not None:
                        np.save(
                            os.path.join(self.feature_dir, 'norm_clstoken', file_name.replace(".h5", ".npy")), 
                            norm_clstoken[j]
                        )
                    
                    # Save patch token features if requested and available
                    if token_features and norm_patchtokens is not None:
                        mean_token = np.zeros((len(patch_markers_batch) * feat_dim))
                        for k, marker_name in enumerate(patch_markers_batch):
                            marker_features = norm_patchtokens[j, k*token_per_marker:(k+1)*token_per_marker, :]
                            mean_token[k*feat_dim:(k+1)*feat_dim] = np.mean(marker_features, axis=0)
                        np.save(
                            os.path.join(self.feature_dir, 'norm_patchtokens', file_name.replace(".h5", ".npy")), 
                            mean_token
                        )
                    
                    processed_patches += 1
        
        print(f"Feature extraction completed! Processed {processed_patches} patches.")
        return processed_patches


class H5ADBuilder:
    """
    A class for building AnnData (h5ad) objects from numpy embedding files.
    
    This class handles loading numpy embedding files, extracting metadata from filenames,
    merging with sample-level metadata, and saving as h5ad objects for downstream analysis.
    
    The class expects embedding files to follow the naming convention:
    {core_name}_{x_coordinate}_{y_coordinate}.npy
    
    Example: A-1_000128_001088.npy where:
    - A-1 is the core name
    - 000128 is the x coordinate
    - 001088 is the y coordinate
    
    Attributes:
        config (dict): Configuration dictionary containing builder parameters.
        embedding_path (str): Directory containing the numpy embedding files.
        output_dir (str): Directory to save the h5ad files.
        metadata_path (str): Path to the metadata CSV file.
        model_name (str): Name of the model used for embeddings.
        patch_size (str): Patch size identifier (e.g., "256_256").
        output_name (str): Name for the output h5ad file.
        metadata (pd.DataFrame): Loaded metadata DataFrame.
    """
    
    def __init__(self, config):
        """
        Initializes the H5ADBuilder class.
        
        Args:
            config (dict): Configuration dictionary containing:
                - embedding_path (str): Directory containing numpy embedding files
                - output_dir (str): Directory to save h5ad files
                - metadata_path (str): Path to metadata CSV file
                - model_name (str): Name of the model (e.g., "Kronos", "Imagenet", "uni")
                - patch_size (str): Patch size identifier (e.g., "256_256", "64_64")
                - output_name (str, optional): Custom output filename. If not provided,
                                             will use format: {dataset}_{patch_size}-{model_name}.h5ad
                - dataset_name (str, optional): Dataset name for default filename (default: "HNSCC")
                - core_id_column (str, optional): Column name for core IDs in metadata (default: "TMA_core_num")
        
        Metadata CSV Format Requirements:
            The metadata CSV file should contain the following structure:
            - First column: Row index (can be unnamed)
            - core_id_column (default: "TMA_core_num"): Core identifier matching the filename prefix
            - Additional columns: Any sample-level metadata (e.g., OS_status, OS..M., Patients, Survival_time)
            
            Example CSV format (based on HNSCC_meta.csv):
            "",TMA_core_num,OS_status,OS..M.,Patients,Survival_time
            "1","I-1",1,4,"I-1","NR"
            "7417","J-1",0,46,"J-1","R"
            "7696","K-1",1,13,"K-1","NR"
            
            Required columns:
            - TMA_core_num (or custom core_id_column): Core identifier (e.g., "A-1", "B-2")
            
            Optional columns (examples from HNSCC dataset):
            - OS_status: Overall survival status (0/1)
            - OS..M.: Overall survival in months
            - Patients: Patient identifier
            - Survival_time: Survival time category ("R"/"NR")
        """
        self.config = config
        self.embedding_path = config["embedding_path"]
        self.output_dir = config["output_dir"]
        self.metadata_path = config["metadata_path"]
        self.model_name = config["model_name"]
        self.patch_size = config["patch_size"]
        self.dataset_name = config.get("dataset_name", "HNSCC")
        self.core_id_column = config.get("core_id_column", "TMA_core_num")
        
        # Set output filename
        if "output_name" in config:
            self.output_name = config["output_name"]
            if not self.output_name.endswith(".h5ad"):
                self.output_name += ".h5ad"
        else:
            self.output_name = f"{self.dataset_name}_adata_{self.patch_size}-{self.model_name}.h5ad"
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load metadata
        self.metadata = self._load_metadata()
    
    def _load_metadata(self):
        """
        Load metadata from CSV file.
        
        Returns:
            pd.DataFrame: Loaded metadata with core IDs as index.
        """
        try:
            print(f"Loading metadata from {self.metadata_path}...")
            metadata = pd.read_csv(self.metadata_path)
            
            # Set the core ID column as index
            if self.core_id_column not in metadata.columns:
                raise ValueError(f"Core ID column '{self.core_id_column}' not found in metadata. "
                               f"Available columns: {list(metadata.columns)}")
            
            metadata.set_index(self.core_id_column, inplace=True)
            print(f"Loaded metadata for {len(metadata)} cores")
            return metadata
            
        except Exception as e:
            print(f"Error loading metadata: {str(e)}")
            sys.exit(1)
    
    def _parse_filename(self, filename):
        """
        Parse filename to extract core name and coordinates.
        
        Args:
            filename (str): Filename to parse (e.g., "A-1_000128_001088.npy")
            
        Returns:
            tuple: (core_name, x_coord, y_coord) or (None, nan, nan) if parsing fails
        """
        try:
            # Remove file extension
            basename = os.path.splitext(filename)[0]
            
            # Split by underscore
            parts = basename.split('_')
            
            if len(parts) >= 3:
                core_name = parts[0]
                x_coord = int(parts[1])
                y_coord = int(parts[2])
                return core_name, x_coord, y_coord
            else:
                print(f"Warning: Could not parse filename {filename} - unexpected format")
                return None, np.nan, np.nan
                
        except Exception as e:
            print(f"Warning: Error parsing filename {filename}: {str(e)}")
            return None, np.nan, np.nan
    
    def build_h5ad(self):
        """
        Build h5ad object from numpy embedding files.
        
        Returns:
            str: Path to the saved h5ad file, or None if no valid embeddings found.
        """
        print(f"\nProcessing {self.model_name} {self.patch_size} embeddings...")
        print(f"Looking for embedding files in: {self.embedding_path}")
        
        # Get all numpy files
        embedding_files = sorted(glob(os.path.join(self.embedding_path, "*.npy")))
        print(f"Found {len(embedding_files)} embedding files")
        
        if len(embedding_files) == 0:
            print(f"No .npy files found in {self.embedding_path}")
            return None
        
        # Process embedding files
        embeddings = []
        core_names = []
        x_coords = []
        y_coords = []
        
        print("Processing embedding files...")
        for i, filepath in enumerate(embedding_files):
            if i % 100 == 0:
                print(f"Processing file {i+1}/{len(embedding_files)}...")
            
            filename = os.path.basename(filepath)
            core_name, x_coord, y_coord = self._parse_filename(filename)
            
            # Only process if we have a valid core name that exists in metadata
            if core_name is not None and core_name in self.metadata.index:
                try:
                    # Load embedding
                    embedding = np.load(filepath)
                    
                    # Ensure 2D array
                    if embedding.ndim == 1:
                        embedding = embedding.reshape(1, -1)
                    
                    embeddings.append(embedding)
                    core_names.append(core_name)
                    x_coords.append(x_coord)
                    y_coords.append(y_coord)
                    
                except Exception as e:
                    print(f"Warning: Error loading {filepath}: {str(e)}")
                    continue
        
        if len(embeddings) == 0:
            print(f"No valid embeddings found for {self.model_name} {self.patch_size}")
            return None
        
        print(f"Successfully loaded {len(embeddings)} valid embeddings")
        
        # Create embedding matrix
        print("Creating embedding matrix...")
        embedding_matrix = np.vstack(embeddings)
        print(f"Embedding matrix shape: {embedding_matrix.shape}")
        
        # Create AnnData object
        print("Creating AnnData object...")
        adata = sc.AnnData(embedding_matrix)
        print(f"AnnData object created with shape: {adata.shape}")
        
        # Add patch-level metadata (coordinates and core info)
        adata.obs[self.core_id_column] = core_names
        adata.obs["x"] = x_coords
        adata.obs["y"] = y_coords
        adata.obs["model"] = self.model_name
        adata.obs["patch_size"] = self.patch_size
        
        # Merge with sample-level metadata
        print("Adding sample-level metadata...")
        adata.obs = adata.obs.merge(
            self.metadata, 
            left_on=self.core_id_column, 
            right_index=True, 
            how="left"
        )
        
        # Check for missing metadata
        missing_metadata = adata.obs[self.core_id_column].isna().sum()
        if missing_metadata > 0:
            print(f"Warning: {missing_metadata} patches have missing metadata")
        
        print("Metadata successfully added")
        
        # Save the AnnData object
        output_path = os.path.join(self.output_dir, self.output_name)
        print(f"Saving AnnData object to {output_path}...")
        adata.write(output_path)
        print("AnnData object saved successfully")
        
        # Print summary
        print(f"\nSummary:")
        print(f"- Total patches: {adata.shape[0]}")
        print(f"- Feature dimensions: {adata.shape[1]}")
        print(f"- Unique cores: {adata.obs[self.core_id_column].nunique()}")
        print(f"- Model: {self.model_name}")
        print(f"- Patch size: {self.patch_size}")
        print(f"- Output saved to: {output_path}")
        
        return output_path


def main():
    """
    Main function demonstrating the usage of all classes.
    """
    # Define root directory and paths
    root_dir = "/fs/ess/PCON0022/Yuzhou/Kronos/HNSCC"
    
    # Configuration for patch extraction (removed mask_dir)
    patch_config = {
        "image_dir": os.path.join(root_dir, "/fs/ess/PCON0022/Yuzhou/Kronos/HNSCC/Datasets/test_TIFF"),
        "output_dir": os.path.join(root_dir, "Datasets", "Patches"),
        "marker_csv_path": os.path.join(root_dir, "Datasets/test_TIFF/", "marker_metadata_HNSCC.csv"),
        "patch_size": 128,  # Using 256x256 as requested
        "stride": 128,      # Non-overlapping patches
        "file_ext": ".tif"
    }
    
    # Configuration for feature extraction - UPDATED for Kronos
    feature_config = {
        "dataset_dir": os.path.join(root_dir, "Datasets", "Patches", "128_128"),
        "feature_dir": os.path.join(root_dir, "Features", "kronos", "128_128", "All_Markers"),
        "checkpoint_path": "hf_hub:MahmoodLab/kronos",  # Path to the pre-trained model checkpoint (Hugging Face Hub)
        "hf_auth_token": None,  # Authentication token for Hugging Face Hub (if checkpoint is from the Hub)
        "cache_dir": os.path.join(root_dir, "Models", "cache"),  # Directory to cache KRONOS model if downloading from Hugging Face Hub
        "model_type": "vits16",  # Type of pre-trained model to use (e.g., vits16)
        "token_overlap": True,  # Whether to use token overlap during feature extraction
        "marker_info": os.path.join(root_dir, "Datasets/test_TIFF/", "marker_metadata_HNSCC.csv"),
        "nuclear_stain": "DAPI",
        "max_value": 65535.0,
        "batch_size": 1,
        "num_workers": 4
    }
    
    # Configuration for H5AD building
    h5ad_config = {
        "embedding_path": os.path.join(root_dir, "Features", "kronos", "128_128", "All_Markers", "norm_patchtokens"),
        "output_dir": os.path.join(root_dir, "AnnData_test"),
        "metadata_path": os.path.join(root_dir, "Datasets", "HNSCC_meta.csv"),
        "model_name": "Kronos",
        "patch_size": "128_128",
        "dataset_name": "HNSCC",
        "core_id_column": "TMA_core_num"
        # "output_name": "custom_name.h5ad"  # Optional: custom output filename
    }
    
    print("=== Patch Extraction Pipeline ===")
    
    # Step 1: Extract patches from entire images
    print("Step 1: Extracting patches from entire images...")
    patch_extractor = PatchExtraction(patch_config)
    
    # Example: Extract patches from specific files
    file_list = ["A-1.tif","L-8.tif"]  # Replace with actual filenames
    # For all files in directory, use: patch_results = patch_extractor.extract_all_patches()
    patch_results = patch_extractor.extract_all_patches(file_list=file_list)
    
    print(f"Patch extraction results: {patch_results}")
    
    print("\n=== Feature Extraction Pipeline ===")
    
    # Step 2: Extract features from patches
    print("Step 2: Extracting features from patches...")
    feature_extractor = FeatureExtraction(feature_config)
    
    # Extract features from all patches
    num_processed = feature_extractor.extract_features_from_patches(token_features=True)
    
    print(f"Feature extraction completed! Processed {num_processed} patches.")
    
    print("\n=== H5AD Building Pipeline ===")
    
    # Step 3: Build h5ad object from embeddings
    print("Step 3: Building h5ad object from embeddings...")
    h5ad_builder = H5ADBuilder(h5ad_config)
    h5ad_path = h5ad_builder.build_h5ad()
    
    if h5ad_path:
        print(f"H5AD object successfully created: {h5ad_path}")
    else:
        print("Failed to create H5AD object")
    
    print("\n=== Pipeline Complete ===")
    print("Patches, features, and h5ad object have been successfully created!")


if __name__ == "__main__":
    main() 