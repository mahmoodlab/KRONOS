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


class PatchDataset(Dataset):
    """
    A PyTorch Dataset class for loading patches and their associated marker data.

    This dataset is used for feature extraction from multiplex images.

    Assumptions:
        - All markers in `marker_list` must exist in the HDF5 files.
        - Marker metadata (mean, std, and ID) is required for normalization.

    Attributes:
        config (dict): Configuration dictionary containing dataset parameters.
        patch_dir (str): Directory containing the patch files (.h5).
        patch_list (list): List of patch file names in the patch directory.
        marker_list (list): List of marker names to be processed.
        marker_max_values (float): Maximum intensity value for marker normalization.
        marker_metadata (pd.DataFrame): Metadata about markers, including marker IDs, means, and standard deviations.
    """

    def __init__(self, config):
        """
        Initializes the PatchDataset.

        Args:
            config (dict): Configuration dictionary containing dataset parameters.
        """
        self.config = config
        self.patch_dir = config["patch_dir"]  # Directory containing patch files
        self.patch_list = os.listdir(self.patch_dir)  # List of patch file names
        self.patch_list = [f for f in self.patch_list if f.endswith(".h5")]
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

        # Return the patch markers, marker IDs, and patch name
        return patch_markers, marker_ids, patch_name

class PatchFeatures(Dataset):
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
        Initializes the PatchFeatures dataset.

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
                - features (np.ndarray): Feature vector for the patch.
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

            n_markers, height, width, feature_dim = features.shape
            features = features.reshape( height * width, n_markers * feature_dim)
            labels = [labels] * features.shape[0]
            
            # Append the flattened feature vector and label to their respective lists
            feature_list.extend(features)
            label_list.extend(labels)

        # Convert the lists to numpy arrays and return them
        return np.array(feature_list), np.array(label_list)
    
class PatchClassifier:
    """
    A classifier for region classifiation and artifact detection using logistic regression.

    This class handles the training, validation, and evaluation of a logistic regression model
    for patch classification based on pre-extracted features. It also uses Optuna for hyperparameter
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
        Initializes the PatchClassifier.

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

        This method initializes the PatchFeatures dataset for each split (train, valid, test),
        retrieves the features and labels, and normalizes the training features.
        """
        if is_train:
            # Load training data and normalize features
            train_obj = PatchFeatures(
                self.config["feature_dir"],
                self.train_df["patch_name"].tolist(),
                self.train_df["label"].tolist()
            )
            self.train_features, self.train_labels = train_obj.get_all()
            self.train_features = self.normalizer.fit_transform(self.train_features)

            # Load validation data and create a DataLoader
            valid_obj = PatchFeatures(
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
        test_obj = PatchFeatures(
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

            batch, n_markers, height, width, feature_dim = features.shape
            
            feat_list = []
            label_list = []
            for i in range(features.shape[0]):
                feat = features[i, :,:,:,:].reshape(height * width, n_markers * feature_dim)
                feat_list.extend(feat)
                label_list.extend([labels[i]] * feat.shape[0])
            features = np.array(feat_list)
            labels = np.array(label_list)
            
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

            batch, n_markers, height, width, feature_dim = features.shape
            
            feat_list = []
            label_list = []
            for i in range(features.shape[0]):
                feat = features[i, :,:,:,:].reshape(height * width, n_markers * feature_dim)
                feat_list.extend(feat)
                label_list.extend([labels[i]] * feat.shape[0])
            features = np.array(feat_list)
            labels = np.array(label_list)

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
        res_dict = {"gt": test_labels, "pred": test_predictions}
        for i in range(n_classes):
            res_dict[f"prob_{i}"] = test_prob[:, i]

        # Save the results to a CSV file
        file_name = "test_results.csv"
        pd.DataFrame(res_dict).to_csv(os.path.join(self.output_dir, file_name), index=False)

class RegionArtifactClassification:
    def __init__(self, config):
        self.config = config

    def generate_dummy_data(self):
        """
        This function will generate dummy images and labels to run this tutorial
        """
        self.image_list = []
        self.labels = []
        # saven random .h5 # files in the image_dir_path
        if not os.path.exists(self.config["patch_dir"]):
            os.makedirs(self.config["patch_dir"], exist_ok=True)
        for i in range(20):
            with h5py.File(os.path.join(self.config["patch_dir"], f"dummy_{i}.h5"), "w") as f:
                for mid, marker_name in enumerate(['DAPI', 'CD11B', 'CD11C']):
                    f.create_dataset(marker_name, data=np.random.rand(256, 256))
                self.image_list.append(f"dummy_{i}.h5")
                self.labels.append(i % 3)

        gt_df = pd.DataFrame({
            "patch_name": self.image_list,
            "label": self.labels
        })
        gt_df.to_csv(os.path.join(self.config["gt_csv_dir"], "patch_annotations.csv"), index=False)
            
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
        dataset = PatchDataset(self.config)
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
                _, _, spatial_features = model(patches_batch, marker_ids=marker_ids_batch)

                # Move the extracted features back to the CPU and convert them to numpy arrays
                spatial_features = spatial_features.cpu().numpy()

                # Save the features for each patch in the batch as a .npy file
                for j, patch_name in enumerate(patch_name_batch):

                    # Replace the .h5 extension with .npy for the feature file
                    patch_name = patch_name.replace(".h5", ".npy")

                    # Save the flattened feature array to the feature directory
                    np.save(os.path.join(feature_dir, f"{patch_name}"), spatial_features[j, :, :, :, :])

    def folds_generation(self):        
        """
        Generates data folds for training, validation, and testing from the cHL dataset.

        This function divides the dataset into four sets based on patch order in the csv file.

        Input:
            - All required parameters, including file paths and patch size, are specified in the configuration dict.

        Output:
            - Each CSV file contains:
                - patch_name: Name of the .h5 file containing the multiplex image patch and
                - label: Patch label 
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

        gt_df = pd.read_csv(os.path.join(self.config["gt_csv_dir"], "patch_annotations.csv"))
        patch_count = gt_df.shape[0]
        folds = []
        folds.append(gt_df[0:patch_count // 4])  # First 25% for fold 1
        folds.append(gt_df[patch_count // 4: patch_count // 2])
        folds.append(gt_df[patch_count // 2: 3 * patch_count // 4])
        folds.append(gt_df[3 * patch_count // 4: patch_count])

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
        2. Train a logistic regression model using Optuna for hyperparameter tuning.
        3. Evaluate the trained model on the test dataset.
        4. Save the results for each fold in the configured result directory.

        Input:
            - All required parameters are specified in the configuration dictionary.

        Output:
            - Trained logistic regression models and evaluation results are saved for each fold.
        """

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

            # Define the output directory for the current fold and create it if it doesn't exist
            output_dir = os.path.join(self.config["result_dir"], fold_id)
            os.makedirs(output_dir, exist_ok=True)

            # Initialize the classifier object for the current fold
            obj = PatchClassifier(self.config, train_df, valid_df, test_df, output_dir)

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

        for fold_id in fold_list:

            # Load the training, validation, and test datasets for the current fold
            test_df = pd.read_csv(os.path.join(self.config["fold_dir"], fold_id, "test.csv"))

            # Define the output directory for the current fold and create it if it doesn't exist
            output_dir = os.path.join(self.config["result_dir"], fold_id)
            os.makedirs(output_dir, exist_ok=True)

            # Initialize the classifier object for the current fold
            obj = PatchClassifier(self.config, None, None, test_df, output_dir)

            # Load the training, validation, and test data
            obj.load_data(is_train=False)

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
            file_name = "test_results.csv"
            test_df = pd.read_csv(os.path.join(self.config["result_dir"], fold_id, file_name))

            # Extract ground truth labels, predictions, and probabilities from the test results
            gt = test_df["gt"].values  # Ground truth labels
            pred = test_df["pred"].values  # Predicted labels
            prob = test_df.iloc[:, 2:].values  # Predicted probabilities for each class

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
        file_name = "average_results.csv"
        results_df.to_csv(os.path.join(self.config["result_dir"], file_name), index=False)
        print("Average results saved successfully")
