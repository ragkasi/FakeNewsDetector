#!/usr/bin/env python3
"""
Convert model.safetensors to pytorch_model.bin for deployment compatibility.
"""

import torch
from safetensors.torch import load_file
import os

def convert_safetensors_to_pytorch(safetensors_path, output_path):
    """
    Convert a safetensors model file to pytorch_model.bin format.
    
    Args:
        safetensors_path (str): Path to the model.safetensors file
        output_path (str): Path where pytorch_model.bin should be saved
    """
    print(f"Loading model from {safetensors_path}...")
    
    # Load the safetensors model
    state_dict = load_file(safetensors_path)
    
    print(f"Model loaded successfully. Found {len(state_dict)} parameters.")
    
    # Save as pytorch_model.bin
    print(f"Saving as pytorch_model.bin to {output_path}...")
    torch.save(state_dict, output_path)
    
    print("Conversion completed successfully!")
    
    # Verify the conversion
    print("Verifying conversion...")
    loaded_state_dict = torch.load(output_path, map_location='cpu')
    
    if len(loaded_state_dict) == len(state_dict):
        print(f"✓ Verification successful! Both models have {len(state_dict)} parameters.")
    else:
        print("✗ Verification failed! Parameter counts don't match.")
        return False
    
    return True

if __name__ == "__main__":
    # Paths
    safetensors_path = "models/bert-fake-news/iteration2/bert-fake-news/checkpoint-5000/model.safetensors"
    output_path = "models/deployment/bert-fake-news/pytorch_model.bin"
    
    # Check if input file exists
    if not os.path.exists(safetensors_path):
        print(f"Error: Input file not found: {safetensors_path}")
        exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Perform conversion
    try:
        success = convert_safetensors_to_pytorch(safetensors_path, output_path)
        if success:
            print(f"\n✓ Model conversion completed!")
            print(f"  Input:  {safetensors_path}")
            print(f"  Output: {output_path}")
        else:
            print("\n✗ Model conversion failed!")
            exit(1)
    except Exception as e:
        print(f"Error during conversion: {e}")
        exit(1) 