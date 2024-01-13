import json
import sys
from model import Model
import torch

def read_config(config_path):
    with open(config_path, 'r') as file:
        return json.load(file)

def process_video(model, config):
    # Process the video using the model and configuration parameters
    result = model.process_controlnet_canny(**config)
    return result

def main(config_path):
    config = read_config(config_path)
    model = Model(device='cuda', dtype=torch.float16)
    result = process_video(model, config)
    # Output the result or save it as needed
    print('Video saved to', result)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <config_file.json>")
        sys.exit(1)
    config_path = sys.argv[1]
    main(config_path)
