import unittest
import torch

def add_parent_to_sys_path():
    import sys
    from pathlib import Path
    current_dir = Path(__file__).resolve()
    parent_dir = current_dir.parent.parent
    sys.path.insert(0, str(parent_dir))

add_parent_to_sys_path()
from kronos import create_model, create_model_from_pretrained


class TestModelLoadingAndInference(unittest.TestCase):
    
    def setUp(self):
        self.device = 'cuda'
        self.batch_size = 2
        self.marker_count = 10
        self.patch_size = 224
        self.batch = torch.randn(self.batch_size, self.marker_count, self.patch_size, self.patch_size).to(self.device)
        self.marker_ids = [torch.tensor([i for i in range(self.marker_count)], device=self.device) for _ in range(self.batch_size)]
        
    def test_model_loading_with_pretrained(self):
        # Test loading with pretrained weights
        model, precision, _ = create_model_from_pretrained(
            checkpoint_path="hf_hub:MahmoodLab/kronos", 
            cache_dir="./model_assets",
        )
        self.assertIsNotNone(model, "Model with pretrained weights failed to load")
        self.precision = precision  

    def test_model_loading_without_pretrained(self):
        # Test loading without pretrained weights
        model_no_weights, precision, _ = create_model(
            checkpoint_path="hf_hub:MahmoodLab/kronos",  
            cache_dir="./model_assets",
        )
        self.assertIsNotNone(model_no_weights, "Model without pretrained weights failed to load")
        self.precision = precision  
    
    def run_inference_and_check_output(self, model):
        # Move model to device and check inference
        model.to(self.device)
        with torch.inference_mode():
            patch_features, marker_features, token_features = model(self.batch, marker_ids=self.marker_ids)
        # Assertions
        self.assertIsNotNone(patch_features, "Model did not produce output")
        self.assertEqual(
            patch_features.shape[0], self.batch_size,
            "Batch size of patch features does not match input batch size"
        )
        self.assertEqual(
            marker_features.shape[1], self.marker_count,
            "Marker count in marker features does not match to input marker count"
        )
        self.assertEqual(
            token_features.shape[1], self.marker_count,
            "Marker count in token features does not match to input marker count"
        )
        
        print(f'Success: Model produced output patch features of shape {patch_features.shape}')
        print(f'Success: Model produced output marker features of shape {marker_features.shape}')
        print(f'Success: Model produced output token features of shape {token_features.shape}')
        

    def test_inference_with_pretrained(self):
        # Test inference for model loaded with pretrained weights
        model, precision, _ = create_model_from_pretrained(
            checkpoint_path="hf_hub:MahmoodLab/kronos",
            cache_dir="./model_assets",
        )
        self.precision = precision
        self.run_inference_and_check_output(model)

    def test_inference_without_pretrained(self):
        # Test inference for model loaded without pretrained weights
        model_no_weights, precision, _ = create_model(
            checkpoint_path="hf_hub:MahmoodLab/kronos",
            cache_dir="./model_assets",
        )
        self.precision = precision
        self.run_inference_and_check_output(model_no_weights)

if __name__ == '__main__':
    unittest.main()