import numpy as np
import pandas as pd
from difflib import SequenceMatcher

class MarkerMetadata:
    """
    A class to handle marker metadata operations, including loading, mapping, and exporting marker information.
    Attributes:
        marker_info (pd.DataFrame): DataFrame containing marker information loaded from a CSV file.
        marker_metatdata (pd.DataFrame): DataFrame containing marker metadata loaded from a CSV file.
        top_suggestions (int): Number of top suggestions to provide for unmatched markers.
        missing_marker_df (pd.DataFrame): DataFrame containing missing markers and their suggestions.
        missing_marker_dict (dict): Dictionary to map missing markers to their resolved metadata.
    Methods:
        __init__(marker_info_csv_path, marker_metadata_csv_path, top_suggestions=5):
            Initializes the MarkerMetadata object by loading marker information and metadata from CSV files.
        get_marker_metadata():
            Matches marker information with metadata, identifies missing markers, and generates suggestions for unmatched markers.
        get_marker_metadata_with_mapping():
            Updates marker information based on user-provided mappings for missing markers.
        set_marker_metadata(marker_metadata_dict):
            Manually sets marker metadata for specific markers and updates the missing marker records.
        export_marker_metadata(output_csv_path):
            Exports the updated marker information to a CSV file.
    """
    def __init__(self, marker_info_csv_path, marker_metadata_csv_path, top_suggestions=5):
        self.marker_info = pd.read_csv(marker_info_csv_path)
        self.marker_metatdata = pd.read_csv(marker_metadata_csv_path)
        self.top_suggestions = top_suggestions
        
    def get_marker_metadata(self):
        self.marker_metatdata.set_index("marker_name", inplace=True)

        self.marker_info["marker_id"] = [0 for i in range(self.marker_info.shape[0])]
        self.marker_info["marker_mean"] = [0.0 for i in range(self.marker_info.shape[0])]
        self.marker_info["marker_std"] = [1.0 for i in range(self.marker_info.shape[0])]

        missing_markers = []
        unmatched_markers = self.marker_metatdata.index.tolist()

        for i, row in self.marker_info.iterrows():
            marker_name = row["marker_name"].upper()
            if marker_name not in self.marker_metatdata.index:
                missing_markers.append(row["marker_name"])
                continue
            unmatched_markers.remove(marker_name)
            self.marker_info.loc[i, "marker_id"] = self.marker_metatdata.loc[marker_name, "marker_id"]
            self.marker_info.loc[i, "marker_mean"] = self.marker_metatdata.loc[marker_name, "marker_mean"]
            self.marker_info.loc[i, "marker_std"] = self.marker_metatdata.loc[marker_name, "marker_std"]

        missing_markers.sort()
        unmatched_markers.sort()

        missing_dict = {"Missing Marker": []}
        for i in range(self.top_suggestions):
            missing_dict[f"Suggestion {i+1}"] = []

        for missing_marker in missing_markers:
            missing_dict["Missing Marker"].append(missing_marker)
            similarity_list = np.array([SequenceMatcher(None, missing_marker.upper(), marker_name).ratio() for marker_name in unmatched_markers])
            sorted_index = np.argsort(similarity_list, stable=True)
            sorted_index = sorted_index[::-1]
            for i in range(self.top_suggestions):
                missing_dict[f"Suggestion {i+1}"].append(unmatched_markers[sorted_index[i]])
        self.missing_marker_df = pd.DataFrame(missing_dict)
        self.missing_marker_df.set_index("Missing Marker", inplace=True)
        
        self.missing_marker_dict = {}
        for missing_marker in missing_markers:
            self.missing_marker_dict[missing_marker] = ''
        return self.marker_metatdata, self.marker_info, self.missing_marker_df, self.missing_marker_dict

    def get_marker_metadata_with_mapping(self):
        matched_markers = []
        for key in self.missing_marker_dict.keys():
            if self.missing_marker_dict[key] == '':
                continue
            
            value = self.missing_marker_dict[key]
            if value in self.marker_metatdata.index:
                self.marker_info.loc[self.marker_info["marker_name"] == key, "marker_id"] = self.marker_metatdata.loc[value, "marker_id"]
                self.marker_info.loc[self.marker_info["marker_name"] == key, "marker_mean"] = self.marker_metatdata.loc[value, "marker_mean"]
                self.marker_info.loc[self.marker_info["marker_name"] == key, "marker_std"] = self.marker_metatdata.loc[value, "marker_std"]
                matched_markers.append(key)
            else:
                print(f"Marker {key} not found in metadata")
        
        for marker_name in matched_markers:
            if marker_name in self.missing_marker_df.index:
                self.missing_marker_df.drop(marker_name, inplace=True)
            del self.missing_marker_dict[marker_name]

    def set_marker_metadata(self, marker_metadata_dict):
        for marker_name in marker_metadata_dict.keys():
            self.marker_info.loc[self.marker_info["marker_name"] == marker_name, "marker_id"] = marker_metadata_dict[marker_name]['marker_id']
            self.marker_info.loc[self.marker_info["marker_name"] == marker_name, "marker_mean"] = marker_metadata_dict[marker_name]['marker_mean']
            self.marker_info.loc[self.marker_info["marker_name"] == marker_name, "marker_std"] = marker_metadata_dict[marker_name]['marker_std']
            if marker_name in self.missing_marker_df.index:
                self.missing_marker_df.drop(marker_name, inplace=True)
            if marker_name in self.missing_marker_dict.keys():
                del self.missing_marker_dict[marker_name]

    def export_marker_metadata(self, output_csv_path):
        self.marker_info = self.marker_info[self.marker_info["marker_id"] != 0]
        self.marker_info.reset_index(drop=True, inplace=True)
        self.marker_info.to_csv(output_csv_path, index=False)
        print(f"Exported marker metadata to {output_csv_path}")
