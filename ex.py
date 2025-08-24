import torch
import os
import sys

def check_model_checkpoint(file_path):
    """
    Loads a PyTorch model checkpoint and checks for NaN or Inf values in its parameters.
    """
    try:
        # Load on CPU to avoid CUDA errors
        checkpoint = torch.load(file_path, map_location='cpu')
        
        # The state dict can be under different keys, so we check common ones.
        state_dict = None
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'model' in checkpoint:
            if hasattr(checkpoint['model'], 'state_dict'):
                 state_dict = checkpoint['model'].state_dict()
            else:
                 state_dict = checkpoint['model']
        else:
            state_dict = checkpoint

        if state_dict is None:
            print(f"Could not find a state_dict in {file_path}")
            return False

        has_nan = False
        has_inf = False

        for param_name, param in state_dict.items():
            if torch.is_tensor(param):
                if torch.isnan(param).any():
                    has_nan = True
                if torch.isinf(param).any():
                    has_inf = True
            if has_nan or has_inf:
                break
        
        if has_nan or has_inf:
            print(f"❌ Found issues in: {file_path}")
            if has_nan:
                print("  - Contains NaN values.")
            if has_inf:
                print("  - Contains Inf values.")
            return False
        else:
            print(f"✅ Checkpoint is clean: {file_path}")
            return True

    except Exception as e:
        print(f"Failed to load or check {file_path}: {e}")
        return False

def find_stable_checkpoints(directory):
    """
    Recursively finds all .pth files in a directory and checks them.
    """
    print(f"Scanning for .pth files in: {directory}")
    stable_checkpoints = []
    for root, _, files in os.walk(directory):
        for file in sorted(files): # Sort for consistent order
            if file.endswith(".pth"):
                file_path = os.path.join(root, file)
                if check_model_checkpoint(file_path):
                    stable_checkpoints.append(file_path)
    
    print("\n--- Summary ---")
    if stable_checkpoints:
        print("Found the following stable checkpoints:")
        for p in stable_checkpoints:
            print(p)
    else:
        print("No stable checkpoints were found in the specified directory.")

if __name__ == "__main__":
    # Directory containing the model checkpoints
    target_directory = "/home/juneyonglee/MyData/5th_years/GOCI_RRS_band3_1day"
    
    if not os.path.isdir(target_directory):
        print(f"Error: Directory not found at '{target_directory}'")
        print("Please make sure the path is correct.")
        sys.exit(1)
        
    find_stable_checkpoints(target_directory)
