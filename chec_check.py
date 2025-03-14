#!/usr/bin/env python
"""
Utility script to check the keys in a PyTorch checkpoint file.
This helps diagnose issues when loading model weights.
"""
import os
import sys
import torch
import argparse
from pprint import pprint

def check_checkpoint(checkpoint_path):
    """
    Load a checkpoint and print its structure and keys.
    
    Args:
        checkpoint_path: Path to the checkpoint file
    """
    print(f"\nExamining checkpoint: {checkpoint_path}")
    
    # Try to load the checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        print("\n✅ Successfully loaded checkpoint")
    except Exception as e:
        print(f"\n❌ Failed to load checkpoint: {e}")
        return
    
    # Check the type
    print(f"\nCheckpoint type: {type(checkpoint)}")
    
    # If it's a dictionary, examine its keys
    if isinstance(checkpoint, dict):
        print("\nTop-level keys:")
        pprint(list(checkpoint.keys()))
        
        # Print information about each key
        print("\nDetails for each key:")
        for k, v in checkpoint.items():
            if isinstance(v, torch.Tensor):
                print(f"  {k}: Tensor of shape {v.shape} and dtype {v.dtype}")
            elif isinstance(v, dict):
                print(f"  {k}: Dictionary with {len(v)} keys")
                # Print a few sample keys if there are many
                if len(v) > 0:
                    sample_keys = list(v.keys())[:3]
                    print(f"    Sample keys: {sample_keys}")
                    if len(v) > 3:
                        print(f"    ... and {len(v) - 3} more keys")
            else:
                print(f"  {k}: {type(v)}")
    else:
        print("Checkpoint is not a dictionary. It's a direct model state_dict.")
        # Try to get a sense of what's in there
        if hasattr(checkpoint, "keys"):
            print("\nTop keys in state_dict:")
            keys_list = list(checkpoint.keys())
            pprint(keys_list[:10])
            if len(keys_list) > 10:
                print(f"... and {len(keys_list) - 10} more keys")
    
    # Suggest how to load the checkpoint
    print("\nSuggested loading approaches:")
    
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            print("  model.load_state_dict(checkpoint['model'])")
        if 'state_dict' in checkpoint:
            print("  model.load_state_dict(checkpoint['state_dict'])")
        if 'model_state_dict' in checkpoint:
            print("  model.load_state_dict(checkpoint['model_state_dict'])")
        if 'ema' in checkpoint:
            print("  model.load_state_dict(checkpoint['ema'])")
            
        if not any(k in checkpoint for k in ['model', 'state_dict', 'model_state_dict', 'ema']):
            # Check if the checkpoint itself might be a state_dict
            likely_state_dict = any('weight' in k or 'bias' in k for k in checkpoint.keys())
            if likely_state_dict:
                print("  model.load_state_dict(checkpoint)  # The checkpoint itself appears to be a state_dict")
    else:
        print("  model.load_state_dict(checkpoint)  # Direct state_dict loading")
    
    print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check the structure of a PyTorch checkpoint")
    # Changed 'data_path' to use 'nargs' and 'default'
    parser.add_argument("checkpoint", nargs="?", default="pretrained/seine.pt", help="Path to the checkpoint file")
    args = parser.parse_args()
    
    check_checkpoint(args.checkpoint)
