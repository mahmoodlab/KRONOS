"""
Patch Phenotyping Utilities
This file provides a comprehensive set of utilities for patch-based phenotyping 
from multiplex images. It includes functionalities for patch extraction, feature 
extraction, dataset preparation, model training, evaluation, and visualization.
Classes:
    PatchPhenotypingDataset:
        Used for patch loading while feature extraction.
    PatchPhenotypingFeatures:
        Used for feature loading while patch phenotyping model training, validation, and testing.
    PatchPhenotypingClassifier:
        A classifier for patch phenotyping using logistic regression. Handles training, 
        validation, and evaluation with hyperparameter optimization using Optuna.
    PatchPhenotyping:
        A high-level class that orchestrates the entire patch phenotyping pipeline, 
        including patch extraction, feature extraction, fold generation, model training, 
        evaluation, and visualization.
"""
import os
import h5py
import optuna
import warnings
import numpy as np
import pandas as pd
import pickle as pkl
from scipy import stats
from matplotlib import pyplot as plt
from sklearn.preprocessing import label_binarize
warnings.filterwarnings('ignore')

from tifffile import TiffFile
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, balanced_accuracy_score, average_precision_score, roc_auc_score

import torch
from torch.utils.data import Dataset
from kronos import create_model_from_pretrained


class PatchPhenotypingDataset(Dataset):
    """
    A PyTorch Dataset class for loading patches and their associated marker data.

    This dataset is used for feature extraction from multiplex images. Each patch contains multiple markers,
    a cell segmentation mask, and a cell type map.

    Assumptions:
        - All markers in `marker_list` must exist in the HDF5 files.
        - Marker metadata (mean, std, and ID) is required for normalization.

    Attributes:
        config (dict): Configuration dictionary containing dataset parameters.
        patch_dir (str): Directory containing the cell-centered patch files (.h5).
        patch_list (list): List of patch file names in the patch directory.
        marker_list (list): List of marker names to be processed.
        marker_max_values (float): Maximum intensity value for marker normalization.
        marker_metadata (pd.DataFrame): Metadata about markers, including marker IDs, means, and standard deviations.
    """

    def __init__(self, config):
        """
        Initializes the CellPhenotypingDataset.

        Args:
            config (dict): Configuration dictionary containing dataset parameters.
        """
        self.config = config
        self.patch_dir = config["patch_dir"]  # Directory containing patch files
        self.patch_list = os.listdir(self.patch_dir)  # List of patch file names
        self.marker_list = config["marker_list"]  # List of marker names to process
        self.marker_max_values = config["marker_max_values"]  # Max intensity value for normalization

        # Load marker metadata from the provided CSV file and set the marker name as the index
        self.marker_metadata = pd.read_csv(config["marker_info_with_metadata_csv_path"])
        self.marker_metadata.set_index("marker_name", inplace=True)

    def __len__(self):
        """
        Returns the total number of patches in the dataset.

        Returns:
            int: Number of patches.
        """
        return len(self.patch_list)

    def __getitem__(self, idx):
        """
        Retrieves a single patch and its associated data.

        Args:
            idx (int): Index of the patch to retrieve.

        Returns:
            tuple: A tuple containing:
                - patch_markers (torch.Tensor): Tensor of shape (num_markers, patch_height, patch_width)
                  containing normalized marker data for the patch.
                - marker_ids (torch.Tensor): Tensor of shape (num_markers,) containing marker IDs.
                - cell_mask (np.ndarray): Binary mask of shape (patch_height, patch_width) indicating the cell region.
                - patch_name (str): Name of the patch file.
        """
        # Get the name and path of the patch file
        patch_name = self.patch_list[idx]
        patch_path = os.path.join(self.patch_dir, patch_name)

        # Open the patch file and read its contents
        with h5py.File(patch_path, "r") as f:

            # Initialize lists to store marker data and IDs
            patch_markers = []
            marker_ids = []

            # Iterate over each marker in the marker list
            for marker_name in self.marker_list:

                # Retrieve marker metadata (ID, mean, and standard deviation)
                marker_id = self.marker_metadata.loc[marker_name, "marker_id"]
                marker_mean = self.marker_metadata.loc[marker_name, "marker_mean"]
                marker_std = self.marker_metadata.loc[marker_name, "marker_std"]

                # Load the marker data and normalize it
                marker = f[marker_name][:] / self.marker_max_values  # Scale by max intensity
                marker = (marker - marker_mean) / marker_std  # Standardize using mean and std

                # Convert the marker data to a PyTorch tensor and append it to the list
                patch_markers.append(torch.tensor(marker))

                # Append the marker ID to the list
                marker_ids.append(np.uint16(marker_id))

            # Stack the marker tensors along the first dimension to create a single tensor
            patch_markers = torch.stack(patch_markers, dim=0)

            # Convert the marker IDs to a PyTorch tensor
            marker_ids = torch.tensor(marker_ids)

        # Return the patch markers, marker IDs, cell mask, and patch name
        return patch_markers, marker_ids, patch_name


class PatchPhenotypingFeatures(Dataset):
    """
    A PyTorch Dataset class for loading pre-extracted features and their associated labels.

    This class is used to handle features extracted from patches for downstream tasks
    such as training, validation, and testing of classification models.

    Attributes:
        feature_dir (str): Directory containing the pre-extracted feature files (.npy).
        patch_list (list): List of patch file names corresponding to the features.
        labels (list): List of labels corresponding to each patch.
    """

    def __init__(self, feature_dir, patch_list, labels):
        """
        Initializes the CellPhenotypingFeatures dataset.

        Args:
            feature_dir (str): Directory containing the pre-extracted feature files (.npy).
            patch_list (list): List of patch file names corresponding to the features.
            labels (list): List of labels corresponding to each patch.
        """
        self.feature_dir = feature_dir  # Directory where feature files are stored
        self.patch_list = patch_list  # List of patch file names
        self.labels = labels  # List of labels for each patch

    def __len__(self):
        """
        Returns the total number of patches in the dataset.

        Returns:
            int: Number of patches.
        """
        return len(self.patch_list)

    def __getitem__(self, idx):
        """
        Retrieves a single feature vector and its associated label.

        Args:
            idx (int): Index of the feature to retrieve.

        Returns:
            tuple: A tuple containing:
                - features (np.ndarray): Flattened feature vector for the patch.
                - labels (int): Label corresponding to the patch.
                - patch_name (str): Name of the patch file.
        """
        # Replace the .h5 extension with .npy to locate the feature file
        patch_name = self.patch_list[idx].replace(".h5", ".npy")
        
        # Load the feature vector from the .npy file and flatten it
        features = np.load(os.path.join(self.feature_dir, patch_name))
        
        # Retrieve the label corresponding to the patch
        labels = self.labels[idx]
            
        return features, labels, patch_name

    def get_all(self):
        """
        Retrieves all feature vectors and their associated labels.

        This method is useful for loading the entire dataset into memory for tasks like training.

        Returns:
            tuple: A tuple containing:
                - feature_list (np.ndarray): Array of all flattened feature vectors.
                - label_list (np.ndarray): Array of all labels.
        """
        feature_list = []  # List to store all feature vectors
        label_list = []  # List to store all labels

        # Iterate through all patches in the dataset
        for i in range(len(self.patch_list)):
            # Replace the .h5 extension with .npy to locate the feature file
            patch_name = self.patch_list[i].replace(".h5", ".npy")
            
            # Load the feature vector from the .npy file
            features = np.load(os.path.join(self.feature_dir, patch_name))
            
            # Retrieve the label corresponding to the patch
            labels = self.labels[i]
            
            # Append the flattened feature vector and label to their respective lists
            feature_list.append(features)
            label_list.append(labels)

        # Convert the lists to numpy arrays and return them
        return np.array(feature_list), np.array(label_list)
    
class PatchPhenotypingClassifier:
    """
    A classifier for cell phenotyping using logistic regression.

    This class handles the training, validation, and evaluation of a logistic regression model
    for cell classification based on pre-extracted features. It also uses Optuna for hyperparameter
    optimization to find the best regularization parameter (C) for the logistic regression model.

    Attributes:
        config (dict): Configuration dictionary containing parameters for training and evaluation.
        train_df (pd.DataFrame): DataFrame containing training data (patch names and labels).
        valid_df (pd.DataFrame): DataFrame containing validation data (patch names and labels).
        test_df (pd.DataFrame): DataFrame containing test data (patch names and labels).
        output_dir (str): Directory to save results, models, and logs.
        normalizer (StandardScaler): Scaler for normalizing feature data.
        model (LogisticRegression): Trained logistic regression model.
    """

    def __init__(self, config, train_df, valid_df, test_df, output_dir):
        """
        Initializes the CellPhenotypingClassifier.

        Args:
            config (dict): Configuration dictionary containing parameters for training and evaluation.
            train_df (pd.DataFrame): DataFrame containing training data (patch names and labels).
            valid_df (pd.DataFrame): DataFrame containing validation data (patch names and labels).
            test_df (pd.DataFrame): DataFrame containing test data (patch names and labels).
            output_dir (str): Directory to save results, models, and logs.
        """
        self.config = config
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        self.output_dir = output_dir
        self.normalizer = StandardScaler()  # Standard scaler for feature normalization
        self.model = None  # Placeholder for the trained logistic regression model

    def load_data(self, is_train=True):
        """
        Loads and preprocesses the training, validation, and test datasets.

        This method initializes the CellPhenotypingFeatures dataset for each split (train, valid, test),
        retrieves the features and labels, and normalizes the training features.
        """
        if is_train:
            # Load training data and normalize features
            train_obj = PatchPhenotypingFeatures(
                self.config["feature_dir"],
                self.train_df["patch_name"].tolist(),
                self.train_df["label"].tolist()
            )
            self.train_features, self.train_labels = train_obj.get_all()
            self.train_features = self.normalizer.fit_transform(self.train_features)

            # Load validation data and create a DataLoader
            valid_obj = PatchPhenotypingFeatures(
                self.config["feature_dir"],
                self.valid_df["patch_name"].tolist(),
                self.valid_df["label"].tolist()
            )
            self.valid_dataloader = torch.utils.data.DataLoader(
                valid_obj,
                batch_size=self.config["feature_batch_size"],
                num_workers=self.config["num_workers"],
                shuffle=False
            )

        # Load test data and create a DataLoader
        test_obj = PatchPhenotypingFeatures(
            self.config["feature_dir"],
            self.test_df["patch_name"].tolist(),
            self.test_df["label"].tolist()
        )
        self.test_dataloader = torch.utils.data.DataLoader(
            test_obj,
            batch_size=self.config["feature_batch_size"],
            num_workers=self.config["num_workers"],
            shuffle=False
        )

    def objective(self, trial):
        """
        Objective function for Optuna hyperparameter optimization.

        This function trains a logistic regression model with a given regularization parameter (C),
        evaluates it on the validation set, and returns the macro-averaged F1-Score.

        Args:
            trial (optuna.trial.Trial): An Optuna trial object for hyperparameter optimization.

        Returns:
            float: Macro-averaged F1-Score on the validation set.
        """
        # Suggest a value for the regularization parameter (C) in the logistic regression model
        C = trial.suggest_float('C', low=self.config["C_low"], high=self.config["C_high"], log=True)

        # Initialize and train the logistic regression model
        model = LogisticRegression(C=1 / C, max_iter=self.config["max_iter"], class_weight='balanced')
        model.fit(self.train_features, self.train_labels)

        # Evaluate the model on the validation set
        valid_labels_list = []
        valid_predictions_list = []
        for features, labels, _ in self.valid_dataloader:
            features = features.numpy()
            labels = labels.numpy()

            # Normalize the features using the training scaler
            features = self.normalizer.transform(features)

            # Predict labels for the validation set
            predictions = model.predict(features)

            # Collect ground truth and predicted labels
            valid_labels_list.extend(labels)
            valid_predictions_list.extend(predictions)

        # Compute the macro-averaged F1-Score
        valid_labels = np.array(valid_labels_list)
        predictions = np.array(valid_predictions_list)
        f1 = f1_score(valid_labels, predictions, average='macro')

        return f1

    def train_and_valid_logistic_regression(self):
        """
        Trains a logistic regression model using Optuna for hyperparameter optimization.

        This method optimizes the regularization parameter (C) using the validation set,
        trains the final model with the best hyperparameters, and saves the model and results.
        """
        # Create an Optuna study for hyperparameter optimization
        study = optuna.create_study(
            study_name="LR Classification",
            direction='maximize',
            sampler=optuna.samplers.TPESampler()
        )
        study.optimize(self.objective, n_trials=self.config['n_trials'])

        # Save the Optuna results to a CSV file
        optuna_dict = {"C": [], "F1-Score": []}
        for trial in study.trials:
            optuna_dict["C"].append(trial.params["C"])
            optuna_dict["F1-Score"].append(trial.value)
        pd.DataFrame(optuna_dict).to_csv(os.path.join(self.output_dir, "optuna_results.csv"), index=False)

        # Print the best results
        print(f'Best F1-Score: {study.best_trial.value}')
        print(f'Best hyperparameters: {study.best_trial.params}')
        best_C = study.best_trial.params['C']

        # Train the final logistic regression model with the best hyperparameters
        self.model = LogisticRegression(C=1 / best_C, max_iter=self.config["max_iter"], class_weight='balanced')
        self.model.fit(self.train_features, self.train_labels)

        # Save the trained model and scaler to a pickle file
        model_dict = {"model": self.model, "norm": self.normalizer, "best_C": best_C}
        pkl.dump(model_dict, open(os.path.join(self.output_dir, "model_dict.pkl"), "wb"))

    def eval(self, label_ratio=0.0):
        """
        Evaluate the trained logistic regression model on the test dataset.

        Args:
            label_ratio (float, optional): A ratio used to modify the output file name.
            If greater than 0, the file name will include the label ratio as a percentage.
            Defaults to 0.0.

        Outputs:
            A CSV file containing the test results, saved in the output directory.
            The file includes ground truth labels, predicted labels, and class probabilities.
        """

        # Load the trained model and scaler from the pickle file
        model_dict = pkl.load(open(os.path.join(self.output_dir, "model_dict.pkl"), "rb"))
        self.model = model_dict["model"]
        self.normalizer = model_dict["norm"]

        test_patch_name_list = []
        test_labels_list = []
        test_predictions_list = []
        test_prob_list = []

        # Iterate through the test DataLoader
        for features, labels, patch_names in self.test_dataloader:
            test_patch_name_list.extend(patch_names)
            features = features.numpy()
            labels = labels.numpy()

            # Normalize the features using the training scaler
            features = self.normalizer.transform(features)

            # Predict labels and probabilities for the test set
            predictions = self.model.predict(features)
            prob = self.model.predict_proba(features)

            # Collect ground truth labels, predictions, and probabilities
            test_labels_list.extend(labels)
            test_predictions_list.extend(predictions)
            test_prob_list.extend(prob)

        # Convert the collected data to numpy arrays
        test_labels = np.array(test_labels_list)
        test_predictions = np.array(test_predictions_list)
        test_prob = np.array(test_prob_list)

        # Prepare the results dictionary
        n_classes = test_prob.shape[1]
        res_dict = {"patch_name": test_patch_name_list, "gt": test_labels, "pred": test_predictions}
        for i in range(n_classes):
            res_dict[f"prob_{i}"] = test_prob[:, i]

        # Save the results to a CSV file
        file_name = f"test_results_{int(label_ratio*100)}.csv" if label_ratio > 0 else "test_results.csv"
        pd.DataFrame(res_dict).to_csv(os.path.join(self.output_dir, file_name), index=False)

 
class PatchPhenotyping:
    def __init__(self, config):
        self.config = config

    def patch_extraction(self):
        """
        Extracts patches of spicific size using sliding window from multiplex images and saves them as HDF5 (.h5) files.

        Assumptions:
            - Patches with no annotated cells are ignored.

        Input:
            - All required parameters are specified in the configuration dict.

        Output:
            - Each patch is saved as an HDF5 file containing:
                - A cell segmentation mask under the key "cell_seg_map".
                - A cell type map under the key "cell_type_map".
                - Individual marker datasets for each marker in the patch.
        """
        
        # Load marker information, which contains metadata about markers in the image
        marker_info_df = pd.read_csv(self.config["marker_info_with_metadata_csv_path"])
        
        # Load ground truth data, which contains cell annotations including cell IDs and their coordinates
        gt_df = pd.read_csv(self.config["gt_csv_path"])
        
        # Retrieve patch size and strid size from the configuration
        patch_size = self.config["patch_size"]
        patch_stride = self.config["patch_stride"]
        
        # Retrieve the directory where patches will be saved and create it if it doesn't exist
        patch_dir = self.config["patch_dir"]
        os.makedirs(patch_dir, exist_ok=True)
        
        # Load the multiplex images from the provided path
        image_list = os.listdir(self.config["image_dir_path"])
        image_list.sort()

        for image_name in image_list:

            multiplex_image = TiffFile(os.path.join(self.config["image_dir_path"], image_name)).asarray()
            
            # Load the cell mask from the provided cell mask directory path
            cell_mask = TiffFile(os.path.join(self.config["cell_mask_dir_path"], image_name)).asarray()
            height, width = cell_mask.shape
            
            # Extract the base name of the multiplex image file (without extension) to use in patch naming
            image_name = image_name.split(".")[0]

            # Filter the ground truth DataFrame to include only the current image
            image_gt_df = gt_df[gt_df["image_name"] == image_name]
            image_gt_df.reset_index(drop=True, inplace=True)
            image_gt_df.set_index("cell_id", inplace=True)

            # Iterate through the multiplex image using a sliding window approach
            for row in range(0, height-patch_size+1, patch_stride):
                for col in range(0, width-patch_size+1, patch_stride):
                    print(f"Extracting patches from {image_name}: {row}/{height} - {col}/{width}", end="\r")
                    patch = multiplex_image[:, row:row+patch_size, col:col+patch_size]
                    patch_cell_mask = cell_mask[row:row+patch_size, col:col+patch_size]
                    unique_cell_ids = np.unique(patch_cell_mask)
                    patch_cell_type = np.zeros_like(patch_cell_mask)-1
                    for cell_id in unique_cell_ids:
                        if cell_id in image_gt_df.index:
                        # Get the cell type id from the ground truth DataFrame
                            cell_type = image_gt_df.loc[cell_id, "cell_type_id"]
                            patch_cell_type[patch_cell_mask == cell_id] = cell_type
                    
                    if np.max(patch_cell_type) < 0:
                        # skipping as no labeled cell is found in the patch
                        continue
                    # Generate a unique name for the patch file using the image name and patch coordinates
                    patch_name = "{}_{:06d}_{:06d}.h5".format(image_name, row, col)

                    # Save the patch and its associated data to an h5 file
                    with h5py.File(os.path.join(patch_dir, patch_name), "w") as f:
                        # Save the cell mask 
                        f.create_dataset("cell_seg_map", data=patch_cell_mask)
                        # Save the cell type map
                        f.create_dataset("cell_type_map", data=patch_cell_type)
                        # Iterate over each marker in the marker information DataFrame
                        for _, r in marker_info_df.iterrows():
                            # Extract the marker name and marker index in the multiplex image
                            marker_name = r["marker_name"]
                            channel_index = r["channel_id"]
                            
                            # Extract the marker-specific patch from the multiplex image
                            marker_patch = patch[channel_index, :, :]
                            
                            # Save the marker patch data in the h5 file under the marker name
                            f.create_dataset(marker_name, data=marker_patch)
            
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
    
    def feature_extraction(self):
        """
        Extracts features from patches using a pre-trained model.

        This function processes patches of multiplex images, applies a pre-trained model to extract features, 
        and saves the resulting features as .npy files for further analysis.

        Steps:
        1. Load the dataset and create a DataLoader for batch processing.
        2. Load the pre-trained model using the KRONOS library.
        3. Iterate through the DataLoader, process each batch, and extract features.
        4. Save the extracted features for each patch as a .npy file.

        Input:
            - All required parameters are specified in the configuration dictionary.

        Output:
            - Features for each patch are saved as .npy files in the configured feature directory.
        """
        # Initialize the dataset and DataLoader for processing patches
        dataset = PatchPhenotypingDataset(self.config)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.config["patch_batch_size"], 
            num_workers=self.config["num_workers"], 
            shuffle=False
        )

        # Load the pre-trained model and retrieve its precision and embedding dimension
        model, _, _ = self.load_model()

        # Create the directory for saving features if it doesn't already exist
        feature_dir = self.config["feature_dir"]
        os.makedirs(feature_dir, exist_ok=True)

        # Set the device to GPU if available, otherwise use CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move the model to the selected device and set it to evaluation mode
        model = model.to(device)
        model.eval()

        # Get the total number of batches for progress tracking
        batch_count = len(dataloader)

        # Disable gradient computation for inference
        with torch.no_grad():
            # Iterate through the DataLoader to process each batch
            for i, (patches_batch, marker_ids_batch, patch_name_batch) in enumerate(dataloader):
                print(f"Processing batch {i+1}/{batch_count}")

                # Move the patches and marker IDs to the selected device
                patches_batch = patches_batch.to(device, dtype=torch.float32)
                marker_ids_batch = marker_ids_batch.to(device, dtype=torch.int64)

                # Pass the patches and marker IDs through the model to extract features
                _, marker_features, _ = model(patches_batch, marker_ids=marker_ids_batch)

                # Move the extracted features back to the CPU and convert them to numpy arrays
                marker_features = marker_features.cpu().numpy()

                # Save the features for each patch in the batch as a .npy file
                for j, patch_name in enumerate(patch_name_batch):

                    # Replace the .h5 extension with .npy for the feature file
                    patch_name = patch_name.replace(".h5", ".npy")

                    # Save the flattened feature array to the feature directory
                    np.save(os.path.join(feature_dir, f"{patch_name}"), marker_features[j, :, :].flatten())

    def folds_generation(self):        
        """
        Generates data folds for training, validation, and testing from the cHL dataset.

        This function divides the dataset into four quadrants based on patch coordinates and 
        uses them to create cross-validation folds. Each fold consists of train.csv, valid.csv, and test.csv files

        This implementation is specific to the cHL dataset but can be extended to other datasets while maintaining 
        the same output format.

        Input:
            - All required parameters, including file paths and patch size, are specified in the configuration dict.

        Output:
            - Each CSV file contains:
                - patch_name: Name of the .h5 file containing the multiplex image patch and cell type map
                - label: Patch label based on majority cell type
                - patch_x: X-coordinate of the patch in the image
                - patch_y: Y-coordinate of the patch in the image
                - label_ratio: Ratio of the majority cell type in the patch 
            - Folds are saved in separate directories under the configured fold directory.
            Example structure:
            ├── fold_1
            │   ├── train.csv
            │   ├── valid.csv
            │   ├── test.csv
            ├── fold_2
            │   ├── train.csv
            │   ├── valid.csv
            │   ├── test.csv
            ...
        """
        
        # Create the directory for storing folds if it doesn't already exist
        os.makedirs(self.config["fold_dir"], exist_ok=True)

        # Get the dimensions of the multiplex image from the TIFF file
        image_list = os.listdir(self.config["image_dir_path"])
        image_path = os.path.join(self.config["image_dir_path"], image_list[0])
        height, width = TiffFile(image_path).pages[0].shape
        
        # Retrieve the patch size from the configuration
        patch_size = self.config["patch_size"]

        # Define the boundaries for the four quadrants of the image
        q1_x1, q1_x2, q1_y1, q1_y2 = 0, width // 2 - patch_size, 0, height // 2 - patch_size  # Top-left quadrant
        q2_x1, q2_x2, q2_y1, q2_y2 = width // 2 + patch_size, width, 0, height // 2 - patch_size  # Top-right quadrant
        q3_x1, q3_x2, q3_y1, q3_y2 = 0, width // 2 - patch_size, height // 2 + patch_size, height  # Bottom-left quadrant
        q4_x1, q4_x2, q4_y1, q4_y2 = width // 2 + patch_size, width, height // 2 + patch_size, height  # Bottom-right quadrant

        # Print the boundaries of each quadrant for debugging purposes
        print(f"Quadrant 1: ({q1_x1}, {q1_y1}) - ({q1_x2}, {q1_y2})")
        print(f"Quadrant 2: ({q2_x1}, {q2_y1}) - ({q2_x2}, {q2_y2})")
        print(f"Quadrant 3: ({q3_x1}, {q3_y1}) - ({q3_x2}, {q3_y2})")
        print(f"Quadrant 4: ({q4_x1}, {q4_y1}) - ({q4_x2}, {q4_y2})")


        data_dict = {"patch_name": [], "label": [], "patch_x": [], "patch_y": [], "label_ratio": []}
        patch_list = os.listdir(self.config["patch_dir"])
        patch_list.sort()

        for patch_name in patch_list:
            cell_type_map = h5py.File(os.path.join(self.config["patch_dir"], patch_name), "r")["cell_type_map"][:]
            cell_types, cell_type_count = stats.mode(cell_type_map.flatten())
            label_ratio = cell_type_count / cell_type_map.flatten().shape[0]
            if isinstance(cell_types, np.ndarray):
                cell_types = cell_types[0]
            else:
                cell_types = cell_types
            
            if cell_types < 0:
                continue

            patch_coords = patch_name.split(".")[0].split("_")[-2:]
            patch_x, patch_y = int(patch_coords[0]), int(patch_coords[1])
            data_dict["patch_name"].append(patch_name)
            data_dict["label"].append(cell_types)
            data_dict["patch_x"].append(patch_x)
            data_dict["patch_y"].append(patch_y)
            data_dict["label_ratio"].append(label_ratio)

        gt_df = pd.DataFrame(data_dict)

        # Split the cells into folds based on their coordinates in the image quadrants
        folds = []
        folds.append(gt_df[(gt_df["patch_x"] >= q1_x1) & (gt_df["patch_x"] <= q1_x2) & (gt_df["patch_y"] >= q1_y1) & (gt_df["patch_y"] <= q1_y2)])
        folds.append(gt_df[(gt_df["patch_x"] >= q2_x1) & (gt_df["patch_x"] <= q2_x2) & (gt_df["patch_y"] >= q2_y1) & (gt_df["patch_y"] <= q2_y2)])
        folds.append(gt_df[(gt_df["patch_x"] >= q3_x1) & (gt_df["patch_x"] <= q3_x2) & (gt_df["patch_y"] >= q3_y1) & (gt_df["patch_y"] <= q3_y2)])
        folds.append(gt_df[(gt_df["patch_x"] >= q4_x1) & (gt_df["patch_x"] <= q4_x2) & (gt_df["patch_y"] >= q4_y1) & (gt_df["patch_y"] <= q4_y2)])

        # Save the folds into separate directories for training, validation, and testing
        for fid in range(1, 5):
            # Create a directory for the current fold
            os.makedirs(f"{self.config['fold_dir']}/fold_{fid}", exist_ok=True)
            
            # Initialize DataFrames for training, validation, and testing
            train_df = None
            valid_df = None
            test_df = None
            
            # Assign folds to training, validation, and testing sets
            for i, fold in enumerate(folds):
                if i == (fid - 1):
                    # The current fold is used as the test set
                    test_df = fold
                else:
                    # The remaining folds are used for training
                    if train_df is None:
                        train_df = fold
                    else:
                        train_df = pd.concat([train_df, fold])
            
            # Randomly sample 20% of the training data for validation
            valid_df = train_df.sample(frac=0.2, random_state=42)
            
            # Remove the validation samples from the training set
            train_df = train_df.drop(valid_df.index)
            
            # Save the training, validation, and testing DataFrames to CSV files
            train_df.to_csv(f"{self.config['fold_dir']}/fold_{fid}/train.csv", index=False)
            valid_df.to_csv(f"{self.config['fold_dir']}/fold_{fid}/valid.csv", index=False)
            test_df.to_csv(f"{self.config['fold_dir']}/fold_{fid}/test.csv", index=False)
            
            # Print a summary of the fold creation process
            print(f"Fold {fid} created successfully")
            print(f"Train: {train_df.shape[0]}, Valid: {valid_df.shape[0]}, Test: {test_df.shape[0]}, Total: {train_df.shape[0] + valid_df.shape[0] + test_df.shape[0]}")

    def train_classification_model(self):
        """
        Trains and evaluates a logistic regression classifier for each fold.

        This function iterates through the cross-validation folds, trains a logistic regression model 
        on the training data, validates it on the validation data, and evaluates it on the test data. 
        The results for each fold are saved in the corresponding output directory.

        Steps:
        1. Load the training, validation, and test datasets for each fold.
        2. Optionally, limit the number of cells per cell type in the training data.
        3. Train a logistic regression model using Optuna for hyperparameter tuning.
        4. Evaluate the trained model on the test dataset.
        5. Save the results for each fold in the configured result directory.

        Input:
            - All required parameters are specified in the configuration dictionary.

        Output:
            - Trained logistic regression models and evaluation results are saved for each fold.
        """
        # Retrieve the maximum number of cells per cell type from the configuration
        max_patches_per_type = self.config["max_patches_per_type"]

        # Get the list of fold directories
        fold_list = os.listdir(self.config["fold_dir"])
        fold_list.sort()

        # Iterate through each fold
        for fold_id in fold_list:
            print("Processing Fold: ", fold_id)

            # Load the training, validation, and test datasets for the current fold
            train_df = pd.read_csv(os.path.join(self.config["fold_dir"], fold_id, "train.csv"))
            valid_df = pd.read_csv(os.path.join(self.config["fold_dir"], fold_id, "valid.csv"))
            test_df = pd.read_csv(os.path.join(self.config["fold_dir"], fold_id, "test.csv"))
            
            # If a maximum number of cells per type is specified, limit the training data
            if max_patches_per_type is not None:
                # Select a maximum of N cells per cell type from the training data
                train_df = train_df.groupby("label").apply(
                    lambda x: x.sample(min(len(x), max_patches_per_type), random_state=42)
                ).reset_index(drop=True)

            # Define the output directory for the current fold and create it if it doesn't exist
            output_dir = os.path.join(self.config["result_dir"], fold_id)
            os.makedirs(output_dir, exist_ok=True)

            # Initialize the classifier object for the current fold
            obj = PatchPhenotypingClassifier(self.config, train_df, valid_df, test_df, output_dir)

            # Load the training, validation, and test data
            obj.load_data()

            # Train the logistic regression model and validate it
            obj.train_and_valid_logistic_regression()

            # Evaluate the trained model on the test dataset
            obj.eval()
        
        # Print a message indicating that training for all folds is complete
        print("Training completed successfully")

    def eval_classification_model(self):
        """
        Evaluates the trained logistic regression model on the test dataset.

        This function computes predictions and probabilities for the test set, saves the results
        to a CSV file, and prepares the data for further analysis.

        Input:
            - All other required parameters are specified in the configuration dictionary.

        Output:
            - Evaluation results are saved in a CSV file for each fold.
        """

        # Get the list of fold directories
        fold_list = os.listdir(self.config["fold_dir"])
        fold_list.sort()


        for label_ratio in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            # Iterate through each fold
            for fold_id in fold_list:

                # Load the training, validation, and test datasets for the current fold
                test_df = pd.read_csv(os.path.join(self.config["fold_dir"], fold_id, "test.csv"))
                test_df = test_df[test_df["label_ratio"]>=label_ratio]
                test_df.reset_index(drop=True, inplace=True)

                # Define the output directory for the current fold and create it if it doesn't exist
                output_dir = os.path.join(self.config["result_dir"], fold_id)
                os.makedirs(output_dir, exist_ok=True)

                # Initialize the classifier object for the current fold
                obj = PatchPhenotypingClassifier(self.config, None, None, test_df, output_dir)

                # Load the training, validation, and test data
                obj.load_data(is_train=False)

                # Evaluate the trained model on the test dataset
                obj.eval(label_ratio=label_ratio)
            
            # Print a message indicating that training for all folds is complete
        print("Evaluation completed successfully")

    def calculate_results(self):
        """
        Calculates and aggregates evaluation metrics across all folds.

        This function reads the test results for each fold, computes evaluation metrics 
        (F1-Score, Balanced Accuracy, Average Precision, and ROC AUC), and calculates 
        the average and standard deviation of these metrics across all folds. The results 
        are saved to a CSV file for further analysis.

        Steps:
        1. Iterate through each fold directory and load the test results.
        2. Compute evaluation metrics for each fold.
        3. Aggregate the metrics across folds to compute the average and standard deviation.
        4. Save the aggregated results to a CSV file.

        Output:
            - A CSV file named "average_results.csv" containing the metrics for each fold, 
              as well as the average and standard deviation across all folds.
        """
        print("Calculating the average results")
        for label_ratio in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            # Initialize a dictionary to store results for each fold
            results_dict = {
                "Fold": [],  # List of fold identifiers
                "F1-Score": [],  # F1-Score for each fold
                "Balanced Accuracy": [],  # Balanced Accuracy for each fold
                "Average Precision": [],  # Average Precision for each fold
                "ROC AUC": []  # ROC AUC for each fold
            }

            # Get the list of fold directories and sort them
            fold_list = os.listdir(self.config["fold_dir"])
            fold_list.sort()

            # Iterate through each fold to compute metrics
            for fold_id in fold_list:
                print(f"Processing Fold: {fold_id}")

                # Load the test results for the current fold
                file_name = f"test_results_{int(label_ratio*100)}.csv" if label_ratio > 0 else "test_results.csv"
                test_df = pd.read_csv(os.path.join(self.config["result_dir"], fold_id, file_name))

                # Extract ground truth labels, predictions, and probabilities from the test results
                gt = test_df["gt"].values  # Ground truth labels
                pred = test_df["pred"].values  # Predicted labels
                prob = test_df.iloc[:, 3:].values  # Predicted probabilities for each class

                # Compute evaluation metrics for the current fold
                f1 = f1_score(gt, pred, average='macro')  # Macro-averaged F1-Score
                bal_acc = balanced_accuracy_score(gt, pred)  # Balanced Accuracy
                if np.unique(gt).shape[0] < (self.config["num_classes"]):
                    # If not all classes are present in the ground truth, adjust the probabilities
                    # avg_prec = 0.0
                    # roc_auc = 0.0
                    gt_bin = label_binarize(gt, classes=np.arange(self.config["num_classes"]))
                    present_classes = np.unique(gt)
                    gt_bin = gt_bin[:, present_classes]  # Binarize the ground truth labels
                    prob = prob[:, present_classes]  # Select only the classes present in the ground truth
                    avg_prec = average_precision_score(gt_bin, prob, average='macro')  # Macro-averaged Average Precision
                    roc_auc = roc_auc_score(gt_bin, prob, average='macro', multi_class='ovr')  # Macro-averaged ROC AUC
                else:
                    avg_prec = average_precision_score(gt, prob, average='macro')  # Macro-averaged Average Precision
                    roc_auc = roc_auc_score(gt, prob, average='macro', multi_class='ovr')  # Macro-averaged ROC AUC

                # Append the metrics to the results dictionary
                results_dict["Fold"].append(fold_id)
                results_dict["F1-Score"].append(f1)
                results_dict["Balanced Accuracy"].append(bal_acc)
                results_dict["Average Precision"].append(avg_prec)
                results_dict["ROC AUC"].append(roc_auc)

            # Compute the average metrics across all folds
            results_dict["Fold"].append("Average")
            results_dict["F1-Score"].append(np.mean(results_dict["F1-Score"]))
            results_dict["Balanced Accuracy"].append(np.mean(results_dict["Balanced Accuracy"]))
            results_dict["Average Precision"].append(np.mean(results_dict["Average Precision"]))
            results_dict["ROC AUC"].append(np.mean(results_dict["ROC AUC"]))

            # Compute the standard deviation of metrics across all folds
            results_dict["Fold"].append("Standard Deviation")
            results_dict["F1-Score"].append(np.std(results_dict["F1-Score"]))
            results_dict["Balanced Accuracy"].append(np.std(results_dict["Balanced Accuracy"]))
            results_dict["Average Precision"].append(np.std(results_dict["Average Precision"]))
            results_dict["ROC AUC"].append(np.std(results_dict["ROC AUC"]))

            # Convert the results dictionary to a DataFrame
            results_df = pd.DataFrame(results_dict)

            # Save the results DataFrame to a CSV file
            file_name = f"average_results_{int(label_ratio*100)}.csv" if label_ratio > 0 else "average_results.csv"
            results_df.to_csv(os.path.join(self.config["result_dir"], file_name), index=False)
        print("Average results saved successfully")

    def plot_results(self):
        """
        Plots and saves the results of model performance metrics across different label ratios.
        This function reads performance metrics (Balanced Accuracy, F1-Score, and Average Precision) 
        from CSV files corresponding to different label ratios. It calculates the mean and standard 
        deviation for each metric, generates bar plots with error bars, and saves the plots as SVG 
        and PNG files.
        The x-axis represents the minimum percentage of the majority class pixels per patch, 
        while the y-axis represents the metric values.
        Saves the plots in the directory specified by `self.config["result_dir"]`.
        
        Attributes:
            config (dict): Configuration dictionary containing paths and parameters:
                - result_dir (str): Directory containing test results.
        Outputs:
            - Saves bar plots for each metric (Balanced Accuracy, F1-Score, Average Precision) 
              with error bars in SVG and PNG formats.
        """
        mean_dict = {"Balanced Accuracy": [], "F1-Score": [], "Average Precision": []}
        std_dict = {"Balanced Accuracy": [], "F1-Score": [], "Average Precision": []}

        label_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for label_ratio in label_ratios:
            file_name = f"average_results_{int(label_ratio*100)}.csv" if label_ratio > 0 else "average_results.csv"
            df = pd.read_csv(os.path.join(self.config["result_dir"], file_name))
            for key in mean_dict.keys():
                mean_dict[key].append(df[key].values[-2])
                std_dict[key].append(df[key].values[-1])
        
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        for i, metric in enumerate(mean_dict.keys()):
            ax = axs[i]
            x = np.arange(len(label_ratios))
            ax.bar(x, mean_dict[metric], yerr=std_dict[metric], label=metric)
            ax.set_ylabel(metric)
            ax.set_ylim([0.0, 1.0])
            ax.set_yticks([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90])
            ax.set_xticks(x, label_ratios)
            if i == 1:
                ax.set_xlabel("Minimum percentage of the majority class pixels per patch")
        plt.show()
        fig.savefig(os.path.join(self.config["result_dir"], "average_results.svg"))
        fig.savefig(os.path.join(self.config["result_dir"], "average_results.png"))

    def visual_results(self):
        """
        Visualizes the results of patch-based phenotyping by generating ground truth 
        and prediction maps for multiplex images and saving them as images. Additionally, 
        it displays a colormap legend for the phenotypes.
        This function processes test results from multiple folds, aggregates them, and 
        maps the predictions and ground truth to their respective spatial locations 
        in the multiplex images. It then generates RGB visualizations for both ground 
        truth and predictions, and saves them as PNG files. A phenotype colormap is 
        also displayed alongside the visualizations.
        Steps:
        1. Load test results from multiple folds and aggregate them into a single DataFrame.
        2. Extract spatial coordinates and image names from patch names.
        3. Load multiplex images and initialize ground truth and prediction maps.
        4. Populate the maps with ground truth and prediction values for each patch.
        5. Generate RGB visualizations for the ground truth and prediction maps.
        6. Save the visualizations as PNG files.
        7. Display the visualizations along with a phenotype colormap legend.
        Attributes:
            class_names (list): List of phenotype class names.
            config (dict): Configuration dictionary containing paths and parameters:
                - result_dir (str): Directory containing test results.
                - image_dir_path (str): Directory containing multiplex images.
                - num_classes (int): Number of phenotype classes.
                - patch_size (int): Size of each patch.
        Outputs:
            - Saves ground truth and prediction RGB maps as PNG files in the result directory.
            - Displays the visualizations and phenotype colormap legend.
        """
        class_names = ['B', 'CD4', 'CD8', 'DC', 'Endothelial', 'Epithelial', 'Lymphatic', 'M1', 'M2', 'Mast', 'Monocyte', 'NK', 'Neutrophil', 'Other', 'TReg', 'Tumor']
        num_classes = self.config["num_classes"]
        cmap = plt.get_cmap("tab20", num_classes)

        # load all test results files
        fold_list = os.listdir(self.config["result_dir"])
        fold_list = [f for f in fold_list if "fold_" in f]
        fold_list.sort()

        result_df = None
        # Iterate through each fold
        for fold_id in fold_list:
            df = pd.read_csv(os.path.join(self.config["result_dir"], fold_id, "test_results.csv"))
            if result_df is None:
                result_df = df
            else:
                result_df = pd.concat([result_df, df])
        result_df.reset_index(drop=True, inplace=True)
        
        result_df["image_name"] = result_df["patch_name"].apply(lambda x: "_".join(x.split(".")[0].split("_")[:-2]))
        result_df["x_coor"] = result_df["patch_name"].apply(lambda x: int(x.split(".")[0].split("_")[-2]))
        result_df["y_coor"] = result_df["patch_name"].apply(lambda x: int(x.split(".")[0].split("_")[-1]))

        # Load the multiplex images from the provided path
        image_list = os.listdir(self.config["image_dir_path"])
        image_list.sort()

        patch_size = self.config["patch_size"]
        for image_name in image_list:
            if not os.path.exists(os.path.join(self.config["result_dir"], image_name.split(".")[0]+"_gt.png")):
                image_path = os.path.join(self.config["image_dir_path"], image_name)
                image_name = image_name.split(".")[0]
                height, width = TiffFile(image_path).pages[0].shape
                result_df_image = result_df[result_df["image_name"] == image_name]
                result_df_image.reset_index(drop=True, inplace=True)
                gt_map = np.zeros((height, width, num_classes+1))
                pred_map = np.zeros((height, width, num_classes+1))
                for i in range(result_df_image.shape[0]):
                    x_coor = result_df_image["x_coor"][i]
                    y_coor = result_df_image["y_coor"][i]
                    pred = result_df_image["pred"][i]+1
                    gt = result_df_image["gt"][i]+1
                    gt_map[x_coor:x_coor+patch_size, y_coor:y_coor+patch_size, gt] += 1
                    pred_map[x_coor:x_coor+patch_size, y_coor:y_coor+patch_size, pred] += 1
                gt_map = np.argmax(gt_map, axis=-1)
                pred_map = np.argmax(pred_map, axis=-1)
                rgb_gt_map = np.zeros((height, width, 3))
                rgb_pred_map = np.zeros((height, width, 3))
                for i in range(1, num_classes+1):
                    for j in range(3):
                        gray_scale_map = rgb_gt_map[:, :, j]
                        gray_scale_map[gt_map == i] = cmap(i-1)[j]*255
                        rgb_gt_map[:, :, j] = gray_scale_map

                        gray_scale_map = rgb_pred_map[:, :, j]
                        gray_scale_map[pred_map == i] = cmap(i-1)[j]*255
                        rgb_pred_map[:, :, j] = gray_scale_map
                rgb_gt_map = rgb_gt_map.astype(np.uint8)
                rgb_pred_map = rgb_pred_map.astype(np.uint8)

                plt.imsave(os.path.join(self.config["result_dir"], image_name.split(".")[0]+"_gt.png"), rgb_gt_map)
                plt.imsave(os.path.join(self.config["result_dir"], image_name.split(".")[0]+"_pred.png"), rgb_pred_map)
            else:
                rgb_gt_map = plt.imread(os.path.join(self.config["result_dir"], image_name.split(".")[0]+"_gt.png"))
                rgb_pred_map = plt.imread(os.path.join(self.config["result_dir"], image_name.split(".")[0]+"_pred.png"))
            
            plt.figure(figsize=(18, 5))
            plt.subplot(1, 3, 1)
            plt.imshow(rgb_gt_map)
            plt.axis("off")
            plt.title("Ground Truth")
            plt.subplot(1, 3, 2)
            plt.imshow(rgb_pred_map)
            plt.axis("off")
            plt.title("Predictions")
            plt.subplot(1, 3, 3)

            phenotype_colors = [cmap(i) for i in range(len(class_names))]
            # Plot the colormap with class names
            plt.subplot(1, 3, 3)
            plt.imshow([[phenotype_colors[i]] for i in range(len(class_names))], extent=[0, 1, 0, len(class_names)])
            plt.xticks([])
            plt.yticks([i+0.5 for i in range(len(class_names))], class_names[::-1])
            plt.title("Phenotypes")
