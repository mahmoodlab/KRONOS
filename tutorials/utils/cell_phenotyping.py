"""
This file provides classes and methods for cell phenotyping tasks, including patch extraction, 
feature extraction, dataset preparation, and classification. It is designed to handle multiplex 
images and cell-centered patches for downstream analysis such as cell classification.
Classes:
    - CellPhenotypingDataset: A PyTorch Dataset for loading cell-centered patches and their associated marker data.
    - CellPhenotypingFeatures: A PyTorch Dataset for loading pre-extracted features and their associated labels.
    - CellPhenotypingClassifier: A logistic regression classifier for cell phenotyping with hyperparameter optimization.
    - CellPhenotyping: A high-level class for managing the entire cell phenotyping pipeline, including patch extraction, 
      feature extraction, fold generation, model training, and evaluation.
"""

import os
import h5py
import optuna
import warnings
import numpy as np
import pandas as pd
import pickle as pkl

warnings.filterwarnings('ignore')

from matplotlib import pyplot as plt
from tifffile import TiffFile
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, balanced_accuracy_score, average_precision_score, roc_auc_score

import torch
from torch.utils.data import Dataset
from kronos import create_model_from_pretrained


class CellPhenotypingDataset(Dataset):
    """
    A PyTorch Dataset class for loading cell-centered patches and their associated marker data.

    This dataset is used for feature extraction from multiplex images. Each patch contains multiple markers,
    a binary cell mask, and metadata about the markers.

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
            # Load the binary cell mask
            cell_mask = f["mask"][:]

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

            # Convert the cell mask to an unsigned 8-bit integer array
            cell_mask = np.uint8(cell_mask)

        # Return the patch markers, marker IDs, cell mask, and patch name
        return patch_markers, marker_ids, cell_mask, patch_name

class CellPhenotypingFeatures(Dataset):
    """
    A PyTorch Dataset class for loading pre-extracted features and their associated labels.

    This class is used to handle features extracted from cell-centered patches for downstream tasks
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
    
class CellPhenotypingClassifier:
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

    def load_data(self):
        """
        Loads and preprocesses the training, validation, and test datasets.

        This method initializes the CellPhenotypingFeatures dataset for each split (train, valid, test),
        retrieves the features and labels, and normalizes the training features.
        """
        if self.train_df is not None:
            # Load training data and normalize features
            train_obj = CellPhenotypingFeatures(
                self.config["feature_dir"],
                self.train_df["patch_name"].tolist(),
                self.train_df["cell_type_id"].tolist()
            )
            self.train_features, self.train_labels = train_obj.get_all()
            self.train_features = self.normalizer.fit_transform(self.train_features)

            # Load validation data and create a DataLoader
            valid_obj = CellPhenotypingFeatures(
                self.config["feature_dir"],
                self.valid_df["patch_name"].tolist(),
                self.valid_df["cell_type_id"].tolist()
            )
            self.valid_dataloader = torch.utils.data.DataLoader(
                valid_obj,
                batch_size=self.config["feature_batch_size"],
                num_workers=self.config["num_workers"],
                shuffle=False
            )

        if self.test_df is not None:
            # Load test data and create a DataLoader
            test_obj = CellPhenotypingFeatures(
                self.config["feature_dir"],
                self.test_df["patch_name"].tolist(),
                self.test_df["cell_type_id"].tolist()
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

    def eval(self):
        """
        Evaluates the trained logistic regression model on the test dataset.

        This method computes predictions and probabilities for the test set, saves the results
        to a CSV file, and prepares the data for further analysis.
        """
        # Load the trained model and scaler from the pickle file
        model_dict = pkl.load(open(os.path.join(self.output_dir, "model_dict.pkl"), "rb"))
        self.model = model_dict["model"]
        self.normalizer = model_dict["norm"]

        test_labels_list = []
        test_predictions_list = []
        test_prob_list = []
        test_patch_name_list = []

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
        pd.DataFrame(res_dict).to_csv(os.path.join(self.output_dir, "test_results.csv"), index=False)

 
class CellPhenotyping:
    def __init__(self, config):
        self.config = config

    def patch_extraction(self):
        """
        Extracts cell-centered patches from multiplex images and saves them as HDF5 (.h5) files.

        Assumptions:
            - Cell IDs in the ground truth annotations must exist in the cell mask.
            - Patches near image boundaries are padded with zeros to maintain the desired size.

        Input:
            - All required parameters are specified in the configuration dict.

        Output:
            - Each patch is saved as an HDF5 file containing:
                - A binary mask under the key "mask".
                - Individual marker datasets for each marker in the patch.
        """
        
        # Load marker information, which contains metadata about markers in the image
        marker_info_df = pd.read_csv(self.config["marker_info_with_metadata_csv_path"])
        
        # Load ground truth data, which contains cell annotations including cell IDs and their coordinates
        gt_df = pd.read_csv(self.config["gt_csv_path"])
        
        # Retrieve patch size from the configuration
        patch_size = self.config["patch_size"]
        
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
            
            # Extract the base name of the multiplex image file (without extension) to use in patch naming
            image_name = image_name.split(".")[0]

            image_gt_df = gt_df[gt_df["image_name"] == image_name]
            image_gt_df.reset_index(drop=True, inplace=True)

            # Iterate over each row in the ground truth DataFrame
            for i, row in image_gt_df.iterrows():
                print(f"Extracting patches from {image_name}: {i+1}/{image_gt_df.shape[0]}", end="\r")

                # Extract cell ID and the x, y coordinates of the cell center
                cell_id = row["cell_id"]
                x, y = row["y_center"], row["x_center"] 
                
                # Calculate the patch boundaries, ensuring they stay within the image dimensions
                x1 = (x - (patch_size // 2)) if (x - (patch_size // 2)) >= 0 else 0
                x2 = (x + (patch_size // 2)) if (x + (patch_size // 2)) < multiplex_image.shape[1] else multiplex_image.shape[1]
                y1 = (y - (patch_size // 2)) if (y - (patch_size // 2)) >= 0 else 0
                y2 = (y + (patch_size // 2)) if (y + (patch_size // 2)) < multiplex_image.shape[2] else multiplex_image.shape[2]

                # Extract the patch from the multiplex image using the calculated boundaries
                patch = multiplex_image[:, x1:x2, y1:y2]
                
                # Extract the corresponding region from the cell mask
                mask = cell_mask[x1:x2, y1:y2]

                # Check if the patch size is correct; if not, then pad the patch with zeros
                if patch.shape[1] != patch_size or patch.shape[2] != patch_size:
                    # Calculate the required padding for each dimension
                    pre_pad_x = 0
                    post_pad_x = 0
                    pre_pad_y = 0
                    post_pad_y = 0

                    if (x - (patch_size // 2)) < 0:
                        pre_pad_x = abs(x - (patch_size // 2))
                    if (x + (patch_size // 2)) >= multiplex_image.shape[1]:
                        post_pad_x = abs((x + (patch_size // 2)) - multiplex_image.shape[1])

                    if (y - (patch_size // 2)) < 0:
                        pre_pad_y = abs(y - (patch_size // 2))
                    if (y + (patch_size // 2)) >= multiplex_image.shape[2]:
                        post_pad_y = abs((y + (patch_size // 2)) - multiplex_image.shape[2])

                    
                    # Pad the patch and mask with zeros to match the desired patch size
                    patch = np.pad(patch, ((0, 0), (pre_pad_x, post_pad_x), (pre_pad_y, post_pad_y)), mode='constant', constant_values=0)
                    mask = np.pad(mask, ((pre_pad_x, post_pad_x), (pre_pad_y, post_pad_y)), mode='constant', constant_values=0)

                    assert patch.shape[1] == patch_size and patch.shape[2] == patch_size, "Patch size mismatch after padding"
                    assert mask.shape[0] == patch_size and mask.shape[1] == patch_size, "Mask size mismatch after padding"

                # Ensure the cell ID exists in the mask; otherwise, raise an error
                assert cell_id in np.unique(mask), "Cell id not found in the mask"

                # Create a binary mask where 1 indicates the cell region and 0 indicates regions not belonging to the cell
                mask = (mask == cell_id).astype(np.uint8)

                # Generate a unique name for the patch file using the image name and cell ID
                patch_name = "{}_{:06d}.h5".format(image_name, cell_id)

                # Save the patch and its associated data to an h5 file
                with h5py.File(os.path.join(patch_dir, patch_name), "w") as f:
                    # Save the binary mask of the cell
                    f.create_dataset("mask", data=mask)
                    
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
        Extracts features from cell-centered patches using a pre-trained model.

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
        dataset = CellPhenotypingDataset(self.config)
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
            for i, (patches_batch, marker_ids_batch, cell_mask_batch, patch_name_batch) in enumerate(dataloader):
                print(f"Processing batch {i+1}/{batch_count}")

                # Apply the cell mask to the patches to mask the non-cell regions
                patches_batch = patches_batch * cell_mask_batch.unsqueeze(1)

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

        This function divides the dataset into four quadrants based on cell coordinates and 
        uses them to create cross-validation folds. Each fold consists of train.csv, valid.csv, and test.csv files

        This implementation is specific to the cHL dataset but can be extended to other datasets while maintaining 
        the same output format.

        Input:
            - All required parameters, including file paths and patch size, are specified in the configuration dict.

        Output:
            - Each CSV file contains:
                - patch_name: Name of the .h5 file containing the multiplex image patch and cell mask
                - cell_type_id: Cell type identifier
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

        # Load the ground truth CSV file containing cell annotations
        gt_df = pd.read_csv(self.config["gt_csv_path"])
        
        # Add a new column to the DataFrame for the patch file names based on cell IDs
        gt_df["patch_name"] = ['{}_{:06d}.h5'.format(row["image_name"], row["cell_id"]) for _, row in gt_df.iterrows()]

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

        # Split the cells into folds based on their coordinates in the image quadrants
        folds = []
        folds.append(gt_df[(gt_df["x_center"] >= q1_x1) & (gt_df["x_center"] <= q1_x2) & (gt_df["y_center"] >= q1_y1) & (gt_df["y_center"] <= q1_y2)])
        folds.append(gt_df[(gt_df["x_center"] >= q2_x1) & (gt_df["x_center"] <= q2_x2) & (gt_df["y_center"] >= q2_y1) & (gt_df["y_center"] <= q2_y2)])
        folds.append(gt_df[(gt_df["x_center"] >= q3_x1) & (gt_df["x_center"] <= q3_x2) & (gt_df["y_center"] >= q3_y1) & (gt_df["y_center"] <= q3_y2)])
        folds.append(gt_df[(gt_df["x_center"] >= q4_x1) & (gt_df["x_center"] <= q4_x2) & (gt_df["y_center"] >= q4_y1) & (gt_df["y_center"] <= q4_y2)])

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
        on the training data and validates it on the validation data. 
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
        max_cells_per_type = self.config["max_cells_per_type"]

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
            if max_cells_per_type is not None:
                # Select a maximum of N cells per cell type from the training data
                train_df = train_df.groupby("cell_type_id").apply(
                    lambda x: x.sample(min(len(x), max_cells_per_type), random_state=42)
                ).reset_index(drop=True)

            # Define the output directory for the current fold and create it if it doesn't exist
            output_dir = os.path.join(self.config["result_dir"], fold_id)
            os.makedirs(output_dir, exist_ok=True)

            # Initialize the classifier object for the current fold
            obj = CellPhenotypingClassifier(self.config, train_df, valid_df, test_df, output_dir)

            # Load the training, validation, and test data
            obj.load_data()

            # Train the logistic regression model and validate it
            obj.train_and_valid_logistic_regression()
        
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

         # Iterate through each fold
        for fold_id in fold_list:
            print("Processing Fold: ", fold_id)

            # Load the training, validation, and test datasets for the current fold
            test_df = pd.read_csv(os.path.join(self.config["fold_dir"], fold_id, "test.csv"))
            
            # Define the output directory for the current fold and create it if it doesn't exist
            output_dir = os.path.join(self.config["result_dir"], fold_id)
            os.makedirs(output_dir, exist_ok=True)

            # Initialize the classifier object for the current fold
            obj = CellPhenotypingClassifier(self.config, None, None, test_df, output_dir)

            # Load the training, validation, and test data
            obj.load_data()

            # Evaluate the trained model on the test dataset
            obj.eval()
        
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
            test_df = pd.read_csv(os.path.join(self.config["result_dir"], fold_id, "test_results.csv"))

            # Extract ground truth labels, predictions, and probabilities from the test results
            gt = test_df["gt"].values  # Ground truth labels
            pred = test_df["pred"].values  # Predicted labels
            prob = test_df.iloc[:, 3:].values  # Predicted probabilities for each class

            # Compute evaluation metrics for the current fold
            f1 = f1_score(gt, pred, average='macro')  # Macro-averaged F1-Score
            bal_acc = balanced_accuracy_score(gt, pred)  # Balanced Accuracy
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
        results_df.to_csv(os.path.join(self.config["result_dir"], "average_results.csv"), index=False)
        print("Average results saved successfully")

    def visual_results(self):
        """
        Visualizes the results of cell-based phenotyping by generating ground truth 
        and prediction maps for multiplex images and saving them as images. Additionally, 
        it displays a colormap legend for the phenotypes.
        This function processes test results from multiple folds, aggregates them, and 
        maps the predictions and ground truth to their respective spatial locations 
        in the multiplex images. It then generates RGB visualizations for both ground 
        truth and predictions, and saves them as PNG files. A phenotype colormap is 
        also displayed alongside the visualizations.
        Steps:
        1. Load test results from multiple folds and aggregate them into a single DataFrame.
        2. Extract cell_id and image name from patch names.
        3. Load multiplex images and initialize ground truth and prediction maps.
        4. Populate the maps with ground truth and prediction values for each cell.
        5. Generate RGB visualizations for the ground truth and prediction maps.
        6. Save the visualizations as PNG files.
        7. Display the visualizations along with a phenotype colormap legend.
        Attributes:
            class_names (list): List of phenotype class names.
            config (dict): Configuration dictionary containing paths and parameters:
                - result_dir (str): Directory containing test results.
                - image_dir_path (str): Directory containing multiplex images.
                - num_classes (int): Number of phenotype classes.
        Outputs:
            - Saves ground truth and prediction RGB maps as PNG files in the result directory.
            - Displays the visualizations and phenotype colormap legend.
        """
        class_names = ['B', 'CD4', 'CD8', 'DC', 'Endothelial', 'Epithelial', 'Lymphatic', 'M1', 'M2', 'Mast', 'Monocyte', 'NK', 'Neutrophil', 'Other', 'TReg', 'Tumor']
        num_classes = self.config["num_classes"]
        cmap = plt.get_cmap("tab20", num_classes)
        phenotype_colors = [cmap(i) for i in range(len(class_names))]

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
        
        result_df["image_name"] = result_df["patch_name"].apply(lambda x: "_".join(x.split(".")[0].split("_")[:-1]))
        result_df["cell_id"] = result_df["patch_name"].apply(lambda x: int(x.split(".")[0].split("_")[-1]))


        # Load the list of multiplex images from the provided path
        image_list = os.listdir(self.config["image_dir_path"])
        image_list.sort()
        
        for image_name in image_list:
            if not os.path.exists(os.path.join(self.config["result_dir"], image_name.split(".")[0]+"_gt.png")):
                # Load the corresponding cell mask from the provided cell mask directory path
                cell_mask = TiffFile(os.path.join(self.config["cell_mask_dir_path"], image_name)).asarray()
                height, width = cell_mask.shape
                
                image_name = image_name.split(".")[0]

                result_df_image = result_df[result_df["image_name"] == image_name]
                result_df_image.reset_index(drop=True, inplace=True)
                result_df_image.set_index("cell_id", inplace=True)

                mask_map = cell_mask.flatten()
                df = pd.DataFrame({"cell_id": mask_map})
                df["gt"] = df["cell_id"].apply(lambda x: result_df_image["gt"][x]+1 if x in result_df_image.index else 0)
                df["pred"] = df["cell_id"].apply(lambda x: result_df_image["pred"][x]+1 if x in result_df_image.index else 0)
                gt_map = df["gt"].values.reshape(height, width)
                pred_map = df["pred"].values.reshape(height, width)

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

            # Plot the colormap with class names
            plt.subplot(1, 3, 3)
            plt.imshow([[phenotype_colors[i]] for i in range(len(class_names))], extent=[0, 1, 0, len(class_names)])
            plt.xticks([])
            plt.yticks([i+0.5 for i in range(len(class_names))], class_names[::-1])
            plt.title("Phenotypes")
            