import subprocess
import yaml

# Paths to different mask directories
mask_paths = [
    "/media/juneyonglee/My Book/data/mask/Test/10",
    "/media/juneyonglee/My Book/data/mask/Test/20",
    "/media/juneyonglee/My Book/data/mask/Test/30",
    "/media/juneyonglee/My Book/data/mask/Test/40",
    "/media/juneyonglee/My Book/data/mask/Test/50"
]


# Path to eval.yaml
eval_config_path = "configs/eval.yaml"


# Load the current eval.yaml configuration
with open(eval_config_path, 'r') as file:
    config = yaml.safe_load(file)

# Iterate over each mask path and run the evaluation
for mask_path in mask_paths:
    # Update the mask_root in the configuration
    config['mask_root'] = mask_path
    
    # Save the updated configuration
    with open(eval_config_path, 'w') as file:
        yaml.dump(config, file)

    # Run the evaluation
    subprocess.run(["python", "model/run.py", "--c", "configs/eval.yaml", "--test"])
