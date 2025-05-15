import torch
from kronos import create_model_from_pretrained

if __name__ == "__main__":

    # Test loading without pretrained weights
    model, precision, embedding_dim = create_model_from_pretrained(
        checkpoint_path="hf_hub:MahmoodLab/kronos",  # or provide a local path
        cfg_path=None, # or provide a local path
        hf_auth_token=None, # provide authentication token for Hugging Face Hub
        cache_dir="./model_assets",
        cfg={"model_type": "vits16", "token_overlap": False} # or provide None if using cfg_path
    )

    print("Model precision: ", precision)
    print("Model embedding dimension: ", embedding_dim)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    batch_size = 2
    marker_count = 56
    patch_size = 256

    # generating a batch with random values
    batch = torch.randn(batch_size, marker_count, patch_size, patch_size, dtype=precision).to(device)

    # generating dummy mean and std values for normalization (see marker_meta.csv for actual mean and std values for each marker ids)
    mean = torch.randn(marker_count, dtype=precision).to(device)
    std = torch.randn(marker_count, dtype=precision).to(device)

    # normalizing the batch
    batch = (batch - mean[None, :, None, None]) / std[None, :, None, None]

    # feature extraction
    with torch.no_grad():
        patch_embeddings, marker_embeddings, token_embeddings = model(batch)
    
    print(f'Patch embeddings: {patch_embeddings.shape}')
    print(f'Marker embeddings: {marker_embeddings.shape}')
    print(f'Token embeddings: {token_embeddings.shape}')

