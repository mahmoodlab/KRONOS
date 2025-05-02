import torch
import json
from typing import Optional, Tuple

from . import vision_transformer as vits

def get_model_config(
    checkpoint_path: Optional[str] = None,
    cfg_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    hf_auth_token: Optional[str] = None,
    cfg: Optional[dict] = None,
) -> dict:
    """
    Generate or load configurations for the Kronos model.
    
    Args:
        checkpoint_path (str, optional): Path to the checkpoint, supports Hugging Face Hub paths prefixed with 'hf_hub:'.
        cfg_path (str, optional): Path to the configuration file. Uses config file on Hugging Face Hub if None and checkpoint_path starts with 'hf_hub:'.
        cache_dir (str, optional): Directory to cache files when downloading from Hugging Face Hub.
        hf_auth_token (str, optional): Authentication token for Hugging Face Hub.
        cfg (dict, optional): Configuration dictionary, if provided then ignore the cfg_path.

    Returns:
        dict: Configuration dictionary containing model settings.
    """
    if cfg is not None:
        # Use provided configuration dictionary
        config = cfg
    elif cfg_path is not None:
        # Load configuration from file
        with open(cfg_path) as f:
            config = json.load(f)
    elif checkpoint_path is not None and checkpoint_path.startswith('hf_hub:'):
        from huggingface_hub import hf_hub_download

        # Download config.json from Hugging Face Hub
        cfg_path = hf_hub_download(
            checkpoint_path[len("hf_hub:"):], 
            cache_dir=cache_dir,
            filename="config.json",
            token=hf_auth_token
        )

        # Load configuration from downloaded file
        with open(cfg_path) as f:
            config = json.load(f)
    else:
        # Default configuration if none provided
        print('No configuration provided. A vit_small model with no token overlap is initialized.')
        config = {
            "model_type": "vits16",
            "token_overlap": False
        }

    return config

def create_model(
    checkpoint_path: Optional[str] = None,
    cfg_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    hf_auth_token: Optional[str] = None,
    cfg: Optional[dict] = None,
) -> Tuple[torch.nn.Module, torch.dtype, int]:
    """
    Creates a Kronos model defined with configuration details in config.
    
    Args:
        checkpoint_path (str, optional): Path to the checkpoint, supports Hugging Face Hub paths prefixed with 'hf_hub:'.
        cfg_path (str, optional): Path to the configuration file. Uses config file on Hugging Face Hub if None and checkpoint_path starts with 'hf_hub:'.
        cache_dir (str, optional): Directory to cache files when downloading from Hugging Face Hub.
        hf_auth_token (str, optional): Authentication token for Hugging Face Hub.
        cfg (dict, optional): Configuration dictionary, if provided then ignore the cfg_path.

    Returns:
        Tuple[torch.nn.Module, torch.dtype, int]: The model, its precision, and embedding dimension.
    """
    
    # Get model configuration
    config = get_model_config(
        checkpoint_path=checkpoint_path,
        cfg_path=cfg_path,
        cache_dir=cache_dir,
        hf_auth_token=hf_auth_token,
        cfg=cfg)

    # Default arguments for vision transformer
    vit_kwargs = dict(
        img_size=224,
        patch_size=16,
        stride_size=16,
        num_markers=512,
        init_values=1.0e-05,
        ffn_layer='mlp',
        block_chunks=4,
        num_register_tokens=16,
    )
    if config["model_type"] in ['vits16', 'vitl16'] and config["token_overlap"]:
        # Adjust stride size if token overlap is enabled
        vit_kwargs['stride_size'] = 8

    model = None
    embedding_dim = None
    if config["model_type"] == 'vits16':
        # Create small vision transformer model
        model = vits.__dict__['vit_small'](**vit_kwargs)
        embedding_dim = 384
    elif config["model_type"] == 'vitl16':
        # Create large vision transformer model
        model = vits.__dict__['vit_large'](**vit_kwargs)
        embedding_dim = 1024
    else:
        # Raise error for unsupported model type
        raise ValueError(f'Unsupported model type: {config["model_type"]}')
    
    return model, torch.float32, embedding_dim


def create_model_from_pretrained(
    checkpoint_path: Optional[str] = None,
    cfg_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    hf_auth_token: Optional[str] = None,
    cfg: Optional[dict] = None,
) -> Tuple[torch.nn.Module, torch.dtype, int]:
    """
    Creates and loads a pretrained Kronos model from the given checkpoint.

    Args:
        checkpoint_path (str, optional): Path to the checkpoint, supports Hugging Face Hub paths prefixed with 'hf_hub:'.
        cfg_path (str, optional): Path to the configuration file. Uses config file on Hugging Face Hub if None and checkpoint_path starts with 'hf_hub:'.
        cache_dir (str, optional): Directory to cache files when downloading from Hugging Face Hub.
        hf_auth_token (str, optional): Authentication token for Hugging Face Hub.
        cfg (dict, optional): Configuration dictionary, if provided then ignore the cfg_path.

    Returns:
        Tuple[torch.nn.Module, torch.dtype, int]: The model, its precision, and embedding dimension.
    """

    # Get model configuration
    config = get_model_config(
        checkpoint_path=checkpoint_path,
        cfg_path=cfg_path,
        cache_dir=cache_dir,
        hf_auth_token=hf_auth_token,
        cfg=cfg)

    # Create model and retrieve precision and embedding dimension
    model, precision, embedding_dim = create_model(
        checkpoint_path=checkpoint_path,
        cfg_path=cfg_path,
        cache_dir=cache_dir,
        hf_auth_token=hf_auth_token,
        cfg=config
    )

    # Load checkpoint if provided
    if checkpoint_path and checkpoint_path.startswith("hf_hub:"):
        from huggingface_hub import hf_hub_download
        if config["model_type"] == 'vits16':
            checkpoint_filename = "kronos_vits16_model.pt"
        elif config["model_type"] == 'vitl16':
            checkpoint_filename = "kronos_vitl16_model.pt"
        else:
            raise ValueError(f'Unsupported model type: {config["model_type"]}')
        
        # Download checkpoint from Hugging Face Hub
        checkpoint_path = hf_hub_download(
            checkpoint_path[len("hf_hub:"):], 
            cache_dir=cache_dir,
            filename=checkpoint_filename,
            token=hf_auth_token
        )

    # Load the state dictionary, removing specific prefixes and entries
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    state_dict = state_dict['teacher']
    state_dict = {k.replace('backbone.', ''): v for k, v in state_dict.items()}
    state_dict = {k: v for k, v in state_dict.items() if 'dino_head' not in k}

    # Load the model state
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(f"\033[92mLoaded model weights from {checkpoint_path}\033[0m")

    return model, precision, embedding_dim
