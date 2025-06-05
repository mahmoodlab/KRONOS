"""
Patient Stratification Utilities
This file provides a comprehensive set of utilities for patient stratification 
based on multiplex images and patient-level metadata. It includes functionalities 
for patch extraction, feature extraction, h5ad building, MIL, dataset preparation, 
model training, evaluation, and visualization focused on AUC metrics.

Classes:
    PatientDataset:
        Used for patient-level data loading and feature aggregation.
    PatientFeatures:
        Used for feature loading while patient stratification model training, validation, and testing.
    BagModel:
        MIL model for bag-level classification using instance aggregation.
    MilDataset:
        Dataset class for MIL data handling.
    PatientStratificationClassifier:
        A classifier for patient stratification using MIL. Handles training, 
        validation, and evaluation with AUC-focused metrics.
    PatientStratification:
        A high-level class that orchestrates the entire patient stratification pipeline, 
        including patch extraction, feature extraction, h5ad building, MIL, fold generation, 
        model training, evaluation, and visualization.
"""
import os
import h5py
import warnings
import numpy as np
import pandas as pd
import pickle as pkl
from scipy import stats
from matplotlib import pyplot as plt
from sklearn.preprocessing import label_binarize
warnings.filterwarnings('ignore')

import umap
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import scanpy as sc
from glob import glob

# Import classes from PatchExtraction_FeatureExtraction_h5ad.py
# We'll include the necessary classes directly here for integration
import csv
import sys
import skimage.io as skio
# from torchvision.transforms import v2


# MIL Classes from mil.py
class BagModel(nn.Module):
    '''
    Model for solving MIL (Multiple Instance Learning) problems.

    Args:
        prepNN: A neural network (subclass of `torch.nn.Module`) for processing input instances before aggregation.
        afterNN: A neural network (subclass of `torch.nn.Module`) for processing the aggregated output and producing the final output.
        aggregation_func: A function for aggregating instance-level outputs into bag-level outputs. 
                          Supports `torch.mean`, `torch.max`, or any function with a `dim` argument.

    Returns:
        Output of the forward function, which can be either the final bag-level output or intermediate outputs for nested bags.
    '''

    def __init__(self, prepNN, afterNN, aggregation_func):
        super().__init__()
        self.prepNN = prepNN  # Neural network for instance-level processing
        self.aggregation_func = aggregation_func  # Aggregation function (e.g., mean or max)
        self.afterNN = afterNN  # Neural network for bag-level processing

    def forward(self, input):
        '''
        Forward pass for the BagModel.

        Args:
            input: A tuple containing:
                - input[0]: Instance-level data (tensor).
                - input[1]: Bag identifiers (tensor).

        Returns:
            Bag-level output or intermediate outputs for nested bags.
        '''
        ids = input[1]
        input = input[0]

        # Ensure bag IDs have the correct shape
        if len(ids.shape) == 1:
            ids.resize_(1, len(ids))

        inner_ids = ids[len(ids) - 1]  # Use the last level of bag IDs
        device = input.device

        # Process instances through the prepNN
        NN_out = self.prepNN(input)

        # Identify unique bags and their counts
        unique, inverse, counts = torch.unique(inner_ids, sorted=True, return_inverse=True, return_counts=True)
        idx = torch.cat([(inverse == x).nonzero()[0] for x in range(len(unique))]).sort()[1]
        bags = unique[idx]
        counts = counts[idx]

        # Initialize output tensor for bag-level representations
        output = torch.empty((len(bags), len(NN_out[0])), device=device)

        # Aggregate instance-level outputs into bag-level outputs
        for i, bag in enumerate(bags):
            output[i] = self.aggregation_func(NN_out[inner_ids == bag], dim=0)

        # Process aggregated outputs through the afterNN
        output = self.afterNN(output)

        # Handle nested bags
        if ids.shape[0] == 1:
            return output
        else:
            ids = ids[:len(ids) - 1]
            mask = torch.empty(0, device=device).long()
            for i in range(len(counts)):
                mask = torch.cat((mask, torch.sum(counts[:i], dtype=torch.int64).reshape(1)))
            return (output, ids[:, mask])


class MilDataset(Dataset):
    '''
    Subclass of `torch.utils.data.Dataset` for handling MIL datasets.

    Args:
        data: Tensor containing instance-level data.
        ids: Tensor containing bag identifiers for each instance.
        labels: Tensor containing bag-level labels.
        normalize: Boolean indicating whether to normalize the data.

    Methods:
        __len__: Returns the number of unique bags in the dataset.
        __getitem__: Returns the data, bag IDs, and label for a specific bag.
        n_features: Returns the number of features in the dataset.
    '''

    def __init__(self, data, ids, labels, normalize=True):
        self.data = data
        self.labels = labels
        self.ids = ids

        # Ensure bag IDs have the correct shape
        if len(ids.shape) == 1:
            ids.resize_(1, len(ids))

        self.bags = torch.unique(self.ids[0])  # Unique bag identifiers

        # Normalize the data if required
        if normalize:
            # Move data to CPU for normalization
            data_cpu = data.cpu()

            # Calculate mean and standard deviation
            mean = data_cpu.mean(dim=0, keepdim=True)
            std = data_cpu.std(dim=0, keepdim=True)
            std[std == 0] = 1  # Prevent division by zero

            # Normalize the data
            self.data = (data_cpu - mean) / std

        # Move data back to GPU if it was originally on GPU
        if data.is_cuda:
            self.data = self.data.cuda()
        else:
            self.data = data

    def __len__(self):
        '''
        Returns the number of unique bags in the dataset.
        '''
        return len(self.bags)

    def __getitem__(self, index):
        '''
        Returns the data, bag IDs, and label for a specific bag.

        Args:
            index: Index of the bag.

        Returns:
            A tuple containing:
                - data: Instances belonging to the bag.
                - bagids: Bag identifiers for the instances.
                - labels: Label for the bag.
        '''
        data = self.data[self.ids[0] == self.bags[index]]
        bagids = self.ids[:, self.ids[0] == self.bags[index]]
        labels = self.labels[index]

        return data, bagids, labels

    def n_features(self):
        '''
        Returns the number of features in the dataset.
        '''
        return self.data.size(1)


def collate(batch):
    '''
    Collate function for combining multiple samples into a batch.

    Args:
        batch: List of samples, where each sample is a tuple (data, bagids, labels).

    Returns:
        A tuple containing:
            - out_data: Concatenated instance-level data.
            - out_bagids: Concatenated bag identifiers.
            - out_labels: Stacked bag-level labels.
    '''
    batch_data = []
    batch_bagids = []
    batch_labels = []

    for sample in batch:
        batch_data.append(sample[0])
        batch_bagids.append(sample[1])
        batch_labels.append(sample[2])

    out_data = torch.cat(batch_data, dim=0)
    out_bagids = torch.cat(batch_bagids, dim=1)
    out_labels = torch.stack(batch_labels)

    return out_data, out_bagids, out_labels


class PatientDataset(Dataset):
    """
    A PyTorch Dataset class for loading patient-level data and aggregating features.

    This dataset is used for patient stratification from cell-level or patch-level features.
    It aggregates features at the patient level and associates them with patient-level labels.

    Attributes:
        config (dict): Configuration dictionary containing dataset parameters.
        patient_data (pd.DataFrame): DataFrame containing patient metadata and feature paths.
        feature_dir (str): Directory containing patient-level feature files.
        patient_list (list): List of patient IDs to be processed.
        aggregation_method (str): Method for feature aggregation ('mean', 'max', 'median').
    """

    def __init__(self, config):
        """
        Initializes the PatientDataset.

        Args:
            config (dict): Configuration dictionary containing dataset parameters.
        """
        self.config = config
        self.patient_data = pd.read_csv(config["patient_metadata_csv_path"])
        self.feature_dir = config["feature_dir"]
        self.patient_list = self.patient_data['patient_id'].unique().tolist()
        self.aggregation_method = config.get("aggregation_method", "mean")

    def __len__(self):
        """
        Returns the total number of patients in the dataset.

        Returns:
            int: Number of patients.
        """
        return len(self.patient_list)

    def __getitem__(self, idx):
        """
        Retrieves aggregated features and label for a single patient.

        Args:
            idx (int): Index of the patient to retrieve.

        Returns:
            tuple: A tuple containing:
                - patient_features (torch.Tensor): Aggregated feature vector for the patient.
                - patient_label (int): Label for the patient.
                - patient_id (str): Patient identifier.
        """
        patient_id = self.patient_list[idx]
        
        # Get patient metadata
        patient_info = self.patient_data[self.patient_data['patient_id'] == patient_id].iloc[0]
        patient_label = patient_info['response']  # Assuming 'response' column exists
        
        # Load patient features
        feature_path = os.path.join(self.feature_dir, f"{patient_id}_features.npy")
        
        if os.path.exists(feature_path):
            features = np.load(feature_path)
            
            # Aggregate features based on specified method
            if self.aggregation_method == "mean":
                aggregated_features = np.mean(features, axis=0)
            elif self.aggregation_method == "max":
                aggregated_features = np.max(features, axis=0)
            elif self.aggregation_method == "median":
                aggregated_features = np.median(features, axis=0)
            else:
                aggregated_features = np.mean(features, axis=0)  # Default to mean
                
        else:
            # If features don't exist, return zeros
            feature_dim = self.config.get("feature_dim", 512)
            aggregated_features = np.zeros(feature_dim)
        
        return torch.tensor(aggregated_features, dtype=torch.float32), patient_label, patient_id


class PatientFeatures(Dataset):
    """
    A PyTorch Dataset class for loading pre-aggregated patient-level features and their associated labels.

    This class is used to handle features already aggregated at the patient level for downstream tasks
    such as training, validation, and testing of stratification models.

    Attributes:
        feature_dir (str): Directory containing the pre-aggregated patient feature files (.npy).
        patient_list (list): List of patient IDs corresponding to the features.
        labels (list): List of labels corresponding to each patient.
    """

    def __init__(self, feature_dir, patient_list, labels):
        """
        Initializes the PatientFeatures dataset.

        Args:
            feature_dir (str): Directory containing the pre-aggregated patient feature files (.npy).
            patient_list (list): List of patient IDs corresponding to the features.
            labels (list): List of labels corresponding to each patient.
        """
        self.feature_dir = feature_dir
        self.patient_list = patient_list
        self.labels = labels

    def __len__(self):
        """
        Returns the total number of patients in the dataset.

        Returns:
            int: Number of patients.
        """
        return len(self.patient_list)

    def __getitem__(self, idx):
        """
        Retrieves a single patient's feature vector and associated label.

        Args:
            idx (int): Index of the patient to retrieve.

        Returns:
            tuple: A tuple containing:
                - features (np.ndarray): Feature vector for the patient.
                - labels (int): Label corresponding to the patient.
                - patient_id (str): Patient identifier.
        """
        patient_id = self.patient_list[idx]
        
        # Load the feature vector from the .npy file
        features = np.load(os.path.join(self.feature_dir, f"{patient_id}_aggregated_features.npy"))
        
        # Retrieve the label corresponding to the patient
        labels = self.labels[idx]
            
        return features, labels, patient_id

    def get_all(self):
        """
        Retrieves all patient features and labels at once.

        Returns:
            tuple: A tuple containing:
                - all_features (np.ndarray): Array of all patient feature vectors.
                - all_labels (np.ndarray): Array of all patient labels.
                - all_patient_ids (list): List of all patient IDs.
        """
        all_features = []
        all_labels = []
        all_patient_ids = []
        
        for idx in range(len(self)):
            features, labels, patient_id = self[idx]
            all_features.append(features)
            all_labels.append(labels)
            all_patient_ids.append(patient_id)
            
        return np.array(all_features), np.array(all_labels), all_patient_ids


class MILDataset(Dataset):
    """
    Dataset for Multiple Instance Learning (MIL) from h5ad files.
    """
    
    def __init__(self, h5ad_path, patient_col='TMA_core_num', label_col='response', feature_type='raw', verbose=True):
        """
        Initialize MIL dataset from h5ad file.
        
        Args:
            h5ad_path (str): Path to h5ad file
            patient_col (str): Column name for patient identifiers
            label_col (str): Column name for labels
            feature_type (str): Type of features to use ('raw', 'harmony', 'pca50', 'pca100', etc.)
            verbose (bool): Whether to print detailed information
        """
        self.adata = sc.read_h5ad(h5ad_path)
        self.patient_col = patient_col
        self.label_col = label_col
        self.feature_type = feature_type
        self.verbose = verbose
        
        # Process features based on feature_type
        self.data = self._process_features()
        
        # Get unique patients
        self.patients = self.adata.obs[patient_col].unique()
        self.patient_labels = {}
        
        # Create patient-label mapping
        for patient in self.patients:
            patient_data = self.adata.obs[self.adata.obs[patient_col] == patient]
            if label_col in patient_data.columns:
                # Use the first non-null label for the patient
                labels = patient_data[label_col].dropna()
                if len(labels) > 0:
                    raw_label = labels.iloc[0]
                    # Simple label conversion: R->1, NR->0, keep numeric as is
                    if raw_label == "R" or raw_label == 1:
                        self.patient_labels[patient] = 1
                    elif raw_label == "NR" or raw_label == 0:
                        self.patient_labels[patient] = 0
                    else:
                        print(f"Warning: Unknown label '{raw_label}' for patient {patient}. Using default label 0.")
                        self.patient_labels[patient] = 0
                else:
                    self.patient_labels[patient] = 0  # Default label
            else:
                self.patient_labels[patient] = 0  # Default label
    
    def _process_features(self):
        """
        Process features based on feature_type.
        
        Returns:
            np.ndarray: Processed feature matrix
        """
        if self.feature_type == 'raw':
            if self.verbose:
                print("Using raw data (adata.X)...")
            return self.adata.X
        elif self.feature_type == 'harmony':
            if self.verbose:
                print("Using Harmony embeddings (adata.obsm['X_pca_harmony'])...")
            return self.adata.obsm["X_pca_harmony"]
        elif self.feature_type.startswith('pca'):
            # Extract the PCA dimension from feature_type
            pca_dim = int(self.feature_type[3:])
            if self.verbose:
                print(f"Computing PCA with {pca_dim} components...")
            
            # Run PCA on the raw data
            pca = PCA(n_components=min(pca_dim, min(self.adata.X.shape)))
            pca_result = pca.fit_transform(self.adata.X)
            
            # Use the PCA result with specified dimensions
            data = pca_result[:, :pca_dim]
            if self.verbose:
                print(f"Explained variance with {pca_dim} PCs: {np.cumsum(pca.explained_variance_ratio_)[pca_dim-1]*100:.2f}%")
            
            return data
        else:
            if self.verbose:
                print(f"Unknown feature_type: {self.feature_type}. Using raw data.")
            return self.adata.X

    def __len__(self):
        return len(self.patients)
    
    def get_mil_dataset(self, normalize_data=False):
        """
        Convert to MilDataset format for training.
        
        Args:
            normalize_data (bool): Whether to normalize the data
            
        Returns:
            MilDataset: Dataset in MIL format
        """
        # Convert data to tensor
        data = torch.tensor(self.data, dtype=torch.float)
        
        # Create patient ID mapping
        unique_patients = {pid: idx for idx, pid in enumerate(self.patients)}
        
        # Create bagids tensor
        patient_ids = self.adata.obs[self.patient_col].values
        bagids = torch.tensor([unique_patients[pid] for pid in patient_ids])
        
        # Create bag labels
        bag_labels = torch.stack([torch.tensor(self.patient_labels[patient], dtype=torch.float) 
                                 for patient in self.patients])
        
        return MilDataset(data, bagids, bag_labels, normalize=normalize_data)


class PatientStratificationClassifier:
    """
    A classifier for patient stratification using MIL focused on AUC evaluation.
    
    This class handles training, validation, and evaluation using only AUC metrics
    with repeated cross-validation similar to the musk_MIL benchmarking approach.
    """

    def __init__(self, config):
        """
        Initializes the PatientStratificationClassifier.

        Args:
            config (dict): Configuration dictionary containing model parameters.
        """
        self.config = config
        self.verbose = config.get("verbose", True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.verbose:
            print(f"Using device: {self.device}")

    def train_and_evaluate_fold(self, dataset, train_indices, test_indices, fold_num, repeat_num):
        """
        Train and evaluate a model for one fold.
        
        Args:
            dataset: MIL dataset
            train_indices: Training indices
            test_indices: Test indices  
            fold_num: Fold number
            repeat_num: Repeat number
            
        Returns:
            dict: Results containing AUC scores
        """
        if self.verbose:
            print(f"\nTraining fold {fold_num + 1}/{self.config.get('n_folds', 5)} of repetition {repeat_num + 1}/{self.config.get('n_repeats', 20)}")
        
        # Create train and test subsets
        train_subset = Subset(dataset, train_indices)
        test_subset = Subset(dataset, test_indices)
        
        # Create data loaders using the MIL collate function
        batch_size = self.config.get("batch_size", 4)
        train_dl = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=False)
        test_dl = DataLoader(test_subset, batch_size=batch_size, shuffle=False, collate_fn=collate, drop_last=False)
        
        # Get input dimension from dataset
        input_dim = dataset.n_features()
        
        # Model parameters from musk script
        n_neurons = self.config.get("n_neurons", 256)
        hidden_layers = self.config.get("hidden_layers", [])
        dropout_rate = self.config.get("dropout_rate", 0.0)
        model_aggregation = self.config.get("model_aggregation", torch.mean)
        
        # Build prepNN (instance-level processing)
        layers = []
        
        # First layer
        layers.append(torch.nn.Linear(input_dim, n_neurons))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Dropout(dropout_rate))
        
        # Add additional hidden layers if specified
        current_size = n_neurons
        if hidden_layers:
            for hidden_size in hidden_layers:
                if hidden_size and hidden_size > 0:
                    layers.append(torch.nn.Linear(current_size, hidden_size))
                    layers.append(torch.nn.ReLU())
                    layers.append(torch.nn.Dropout(dropout_rate))
                    current_size = hidden_size
        
        prepNN = torch.nn.Sequential(*layers)
        
        # Build afterNN (bag-level processing)
        afterNN = torch.nn.Sequential(
            torch.nn.Linear(current_size, 1)
        )
        
        # Create BagModel
        model = BagModel(prepNN, afterNN, model_aggregation)
        model = model.to(self.device)
        
        # Training setup
        lr = self.config.get("lr", 1e-3)
        weight_decay = self.config.get("weight_decay", 1e-5)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.BCEWithLogitsLoss()
        
        # Learning rate scheduler
        lr_scheduler = self.config.get("lr_scheduler", True)
        lr_step_size = self.config.get("lr_step_size", 50)
        lr_gamma = self.config.get("lr_gamma", 0.5)
        scheduler = None
        if lr_scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
        
        # Training loop
        n_epochs = self.config.get("n_epochs", 50)
        loss_log_interval = self.config.get("loss_log_interval", 10)
        
        for epoch in range(n_epochs):
            model.train()
            total_loss = 0
            batch_count = 0
            
            for data_batch, bagids_batch, labels_batch in train_dl:
                try:
                    data_batch = data_batch.to(self.device)
                    bagids_batch = bagids_batch.to(self.device)
                    labels_batch = labels_batch.to(self.device)
                    
                    pred = model((data_batch, bagids_batch)).squeeze()
                    
                    # Ensure pred and labels_batch have compatible shapes
                    if pred.ndim == 0:
                        pred = pred.unsqueeze(0)
                    
                    loss = criterion(pred, labels_batch)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    batch_count += 1
                except Exception as e:
                    if self.verbose:
                        print(f"Error in batch: {e}")
                    continue
            
            # Update learning rate if scheduler is enabled
            if scheduler is not None:
                scheduler.step()
            
            if self.verbose and (epoch + 1) % loss_log_interval == 0:
                avg_loss = total_loss / batch_count if batch_count > 0 else float('nan')
                current_lr = scheduler.get_last_lr()[0] if scheduler is not None else lr
                print(f'Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}, LR: {current_lr:.6f}')
        
        # Evaluation
        model.eval()
        test_probs = []
        test_labels = []
        
        with torch.no_grad():
            for data_batch, bagids_batch, labels_batch in test_dl:
                try:
                    data_batch = data_batch.to(self.device)
                    bagids_batch = bagids_batch.to(self.device)
                    labels_batch = labels_batch.to(self.device)
                    
                    pred_raw = model((data_batch, bagids_batch)).squeeze()
                    if pred_raw.ndim == 0:
                        pred_raw = pred_raw.unsqueeze(0)
                    
                    # Apply sigmoid to get probabilities
                    pred_prob = torch.sigmoid(pred_raw)
                    
                    pred_prob_np = pred_prob.detach().cpu().numpy()
                    labels_batch_np = labels_batch.detach().cpu().numpy()
                    
                    test_probs.extend(pred_prob_np)
                    test_labels.extend(labels_batch_np)
                except Exception as e:
                    if self.verbose:
                        print(f"Error in test batch: {e}")
                    continue
        
        # Calculate AUC
        test_auc = auc(*roc_curve(test_labels, test_probs)[:2])
        
        if self.verbose:
            print(f'Fold {fold_num + 1} Testing AUC: {test_auc:.3f}')
        
        return {
            'repeat': repeat_num + 1,
            'fold': fold_num + 1,
            'test_auc': test_auc
        }

    def run_cross_validation(self, h5ad_path, feature_type='raw'):
        """
        Run repeated cross-validation with AUC evaluation.
        
        Args:
            h5ad_path (str): Path to h5ad file
            feature_type (str): Type of features to use
            
        Returns:
            pd.DataFrame: Results dataframe with AUC scores
        """
        if self.verbose:
            print(f"Starting cross-validation with feature type: {feature_type}")
        
        # Create MIL dataset
        mil_dataset_wrapper = MILDataset(h5ad_path, 
                                       patient_col=self.config.get("patient_col", "TMA_core_num"),
                                       label_col=self.config.get("label_col", "response"),
                                       feature_type=feature_type,
                                       verbose=self.verbose)
        
        # Convert to MilDataset format
        normalize_data = self.config.get("normalize_data", False)
        dataset = mil_dataset_wrapper.get_mil_dataset(normalize_data=normalize_data)
        
        # Get bag labels for stratification
        bag_labels = dataset.labels.cpu().numpy()
        
        # Set seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Parameters
        n_repeats = self.config.get("n_repeats", 20)
        n_folds = self.config.get("n_folds", 5)
        
        all_fold_results = []
        
        # Perform repeated cross-validation
        for repeat in range(n_repeats):
            if self.verbose:
                print(f"\n======= Starting repetition {repeat + 1}/{n_repeats} =======")
            
            # Set a different seed for each repetition
            repeat_seed = 42 + repeat
            torch.manual_seed(repeat_seed)
            np.random.seed(repeat_seed)
            
            # Initialize stratified k-fold cross validation
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=repeat_seed)
            
            # Perform k-fold cross validation
            for fold, (train_indices, test_indices) in enumerate(kf.split(np.arange(len(dataset)), bag_labels)):
                fold_result = self.train_and_evaluate_fold(dataset, train_indices, test_indices, fold, repeat)
                if fold_result is not None:
                    all_fold_results.append(fold_result)
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(all_fold_results)
        
        return results_df

    def save_results_and_plot(self, results_df, feature_type, output_dir):
        """
        Save results without creating visualization plots.
        
        Args:
            results_df (pd.DataFrame): Results dataframe
            feature_type (str): Feature type used
            output_dir (str): Output directory
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save per-fold results
        base_name = f"patient_stratification_{feature_type}"
        fold_csv_path = os.path.join(output_dir, f'{base_name}_fold_auc.csv')
        results_df.to_csv(fold_csv_path, index=False)
        if self.verbose:
            print(f"Per-fold AUC results saved to '{fold_csv_path}'")
        
        # Calculate summary statistics
        mean_auc = results_df['test_auc'].mean()
        std_auc = results_df['test_auc'].std()
        n_samples = len(results_df)
        
        # Calculate 95% confidence interval
        confidence_level = 0.95
        alpha = 1 - confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, n_samples - 1)
        margin_error = t_critical * (std_auc / np.sqrt(n_samples))
        ci_lower = mean_auc - margin_error
        ci_upper = mean_auc + margin_error
        
        # Save summary statistics
        summary_stats = {
            'feature_type': feature_type,
            'mean_auc': mean_auc,
            'std_auc': std_auc,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_samples': n_samples
        }
        
        summary_df = pd.DataFrame([summary_stats])
        summary_csv_path = os.path.join(output_dir, f'{base_name}_summary.csv')
        summary_df.to_csv(summary_csv_path, index=False)
        
        if self.verbose:
            print(f"\nSummary Statistics for {feature_type}:")
            print(f"  Mean AUC: {mean_auc:.3f} ± {std_auc:.3f}")
            print(f"  95% CI: [{ci_lower:.3f} - {ci_upper:.3f}]")
            print(f"  Number of samples: {n_samples}")
        
        return summary_stats


class PatientStratification:
    """
    A high-level class that orchestrates the entire patient stratification pipeline.
    
    This class manages the complete workflow from patch extraction through MIL to 
    final AUC visualization results.
    """

    def __init__(self, config):
        """
        Initializes the PatientStratification pipeline.

        Args:
            config (dict): Configuration dictionary containing all parameters.
        """
        self.config = config
        self.verbose = config.get("verbose", True)
        
        # Import the necessary classes for the full pipeline
        try:
            from .PatchExtraction_FeatureExtraction_h5ad import (
                PatchExtraction, FeatureExtraction, H5ADBuilder
            )
            self.PatchExtraction = PatchExtraction
            self.FeatureExtraction = FeatureExtraction  
            self.H5ADBuilder = H5ADBuilder
        except ImportError:
            print("Warning: Could not import patch extraction classes. Pipeline steps 1-3 will be skipped.")
            self.PatchExtraction = None
            self.FeatureExtraction = None
            self.H5ADBuilder = None

    def run_patch_extraction(self):
        """
        Run patch extraction step.
        
        Returns:
            dict: Results from patch extraction
        """
        if self.PatchExtraction is None:
            if self.verbose:
                print("Patch extraction not available. Skipping...")
            return None
            
        if self.verbose:
            print("=== Step 1: Patch Extraction ===")
        
        patch_config = self.config.get("patch_extraction", {})
        extractor = self.PatchExtraction(patch_config)
        
        file_list = patch_config.get("file_list", None)
        results = extractor.extract_all_patches(file_list=file_list)
        
        if self.verbose:
            print(f"Patch extraction completed: {results}")
        return results

    def run_feature_extraction(self):
        """
        Run feature extraction step.
        
        Returns:
            int: Number of patches processed
        """
        if self.FeatureExtraction is None:
            if self.verbose:
                print("Feature extraction not available. Skipping...")
            return None
            
        if self.verbose:
            print("=== Step 2: Feature Extraction ===")
        
        feature_config = self.config.get("feature_extraction", {})
        extractor = self.FeatureExtraction(feature_config)
        
        num_processed = extractor.extract_features_from_patches(
            token_features=feature_config.get("token_features", True)
        )
        
        if self.verbose:
            print(f"Feature extraction completed: {num_processed} patches processed")
        return num_processed

    def run_h5ad_building(self):
        """
        Run h5ad building step.
        
        Returns:
            str: Path to created h5ad file
        """
        if self.H5ADBuilder is None:
            if self.verbose:
                print("H5AD building not available. Skipping...")
            return None
            
        if self.verbose:
            print("=== Step 3: H5AD Building ===")
        
        h5ad_config = self.config.get("h5ad_building", {})
        builder = self.H5ADBuilder(h5ad_config)
        
        h5ad_path = builder.build_h5ad()
        
        if h5ad_path:
            if self.verbose:
                print(f"H5AD building completed: {h5ad_path}")
            # Update config with h5ad path for downstream steps
            self.config["h5ad_path"] = h5ad_path
        else:
            raise ValueError("H5AD building failed")
        
        return h5ad_path

    def run_mil_benchmarking(self, feature_types=None):
        """
        Run MIL benchmarking with different feature types and PCA dimensions.
        
        Args:
            feature_types (list): List of feature types to test (e.g., ['raw', 'pca50', 'pca100', 'harmony'])
            
        Returns:
            dict: Benchmarking results for all feature types
        """
        if self.verbose:
            print("=== Step 4: MIL Benchmarking ===")
        
        if feature_types is None:
            feature_types = self.config.get("feature_types", ['raw', 'pca50', 'pca100'])
        
        h5ad_path = self.config.get("h5ad_path")
        if not h5ad_path or not os.path.exists(h5ad_path):
            raise ValueError(f"H5AD file not found: {h5ad_path}")
        
        output_dir = self.config.get("output_dir", "./results")
        
        # Initialize classifier
        classifier = PatientStratificationClassifier(self.config)
        
        all_results = {}
        summary_stats = []
        
        # Run benchmarking for each feature type
        for feature_type in feature_types:
            if self.verbose:
                print(f"\n--- Benchmarking with {feature_type} features ---")
            
            # Run cross-validation
            results_df = classifier.run_cross_validation(h5ad_path, feature_type)
            
            # Save results and create plots
            summary = classifier.save_results_and_plot(results_df, feature_type, output_dir)
            
            all_results[feature_type] = results_df
            summary_stats.append(summary)
        
        # Create combined summary
        combined_summary = pd.DataFrame(summary_stats)
        combined_csv_path = os.path.join(output_dir, "combined_feature_comparison.csv")
        combined_summary.to_csv(combined_csv_path, index=False)
        
        # Create comparison visualization
        self._create_comparison_plot(combined_summary, output_dir)
        
        if self.verbose:
            print(f"\nBenchmarking completed. Combined results saved to {output_dir}")
        return all_results

    def _create_comparison_plot(self, summary_df, output_dir):
        """
        Create comparison plot for different feature types.
        
        Args:
            summary_df (pd.DataFrame): Summary statistics for all feature types
            output_dir (str): Output directory
        """
        plt.figure(figsize=(12, 8))
        
        feature_types = summary_df['feature_type'].tolist()
        mean_aucs = summary_df['mean_auc'].tolist()
        ci_lowers = summary_df['ci_lower'].tolist()
        ci_uppers = summary_df['ci_upper'].tolist()
        
        x_pos = np.arange(len(feature_types))
        
        # Create bar plot with error bars
        bars = plt.bar(x_pos, mean_aucs, alpha=0.7, capsize=5)
        
        # Add error bars for 95% CI
        plt.errorbar(x_pos, mean_aucs, 
                    yerr=[np.array(mean_aucs) - np.array(ci_lowers),
                          np.array(ci_uppers) - np.array(mean_aucs)],
                    fmt='none', color='black', capsize=5, capthick=2)
        
        # Customize plot
        plt.xlabel('Feature Type')
        plt.ylabel('Mean AUC Score')
        plt.title('Patient Stratification Performance Comparison\n(Mean AUC with 95% Confidence Intervals)')
        plt.xticks(x_pos, feature_types, rotation=45)
        
        # Add value labels on bars
        for i, (mean_auc, ci_lower, ci_upper) in enumerate(zip(mean_aucs, ci_lowers, ci_uppers)):
            plt.text(i, mean_auc + 0.01, f'{mean_auc:.3f}\n[{ci_lower:.3f}, {ci_upper:.3f}]', 
                    ha='center', va='bottom', fontsize=9)
        
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        # Save plot
        comparison_plot_path = os.path.join(output_dir, "feature_type_comparison.png")
        plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        if self.verbose:
            print(f"Feature comparison plot saved to '{comparison_plot_path}'")

    def run_complete_pipeline(self, feature_types=None):
        """
        Run the complete pipeline from patch extraction to MIL benchmarking.
        
        Args:
            feature_types (list): List of feature types to benchmark
            
        Returns:
            dict: Complete pipeline results
        """
        if self.verbose:
            print("=== Running Complete Patient Stratification Pipeline ===")
        
        pipeline_results = {}
        
        # Step 1: Patch Extraction
        if self.config.get("run_patch_extraction", False):
            pipeline_results['patch_extraction'] = self.run_patch_extraction()
        
        # Step 2: Feature Extraction  
        if self.config.get("run_feature_extraction", False):
            pipeline_results['feature_extraction'] = self.run_feature_extraction()
        
        # Step 3: H5AD Building
        if self.config.get("run_h5ad_building", False):
            pipeline_results['h5ad_building'] = self.run_h5ad_building()
        
        # Step 4: MIL Benchmarking (main focus)
        if self.config.get("run_mil_benchmarking", True):
            pipeline_results['mil_benchmarking'] = self.run_mil_benchmarking(feature_types)
        
        if self.verbose:
            print("=== Complete Pipeline Finished ===")
            print(f"Results ready for analysis in: {self.config.get('output_dir', './results')}")
        
        return pipeline_results


def main():
    """
    Example usage of the PatientStratification pipeline.
    """
    # Configuration example with musk parameters
    config = {
        # H5AD file path (required for MIL benchmarking)
        "h5ad_path": "/fs/ess/PAS2205/Yuzhou/Datasets/CTCL_pembro_data/CTCL_adata_256_256-Kronos.h5ad",
        
        # Output directory
        "output_dir": "./patient_stratification_results_test",
        
        # Verbose control - set to False for minimal output
        "verbose": False,  # Set to False to reduce output
        
        # MIL model parameters (from musk script)
        "n_neurons": 256,
        "hidden_layers": [],  # Additional hidden layers
        "dropout_rate": 0.0,
        "weight_decay": 1e-5,
        "lr": 1e-3,
        "lr_scheduler": True,
        "lr_step_size": 50,
        "lr_gamma": 0.5,
        "n_epochs": 50,
        "batch_size": 4,
        "model_aggregation": torch.mean,
        "normalize_data": False,
        "loss_log_interval": 10,
        
        # Cross-validation parameters
        "n_repeats": 20,
        "n_folds": 5,
        
        # Data parameters
        "patient_col": "Patients",
        "label_col": "Response",
        
        # Feature types to benchmark
        "feature_types": ['pca50', 'pca100'],
        
        # Pipeline control (set to False if you already have h5ad file)
        "run_patch_extraction": False,
        "run_feature_extraction": False,
        "run_h5ad_building": False,
        "run_mil_benchmarking": True,
    }
    
    # Run pipeline
    stratification = PatientStratification(config)
    results = stratification.run_complete_pipeline()
    
    if config.get("verbose", True):
        print("Pipeline completed successfully!")


if __name__ == "__main__":
    main()