import json
import sys
import torch
from model import Model

def read_config(config_path):
    with open(config_path, 'r') as file:
        return json.load(file)

def process_video(model, config):
    result = model.process_pix2pix(**config)
    return result

def main(config_path):
    config = read_config(config_path)
    model = Model(device='cuda', dtype=torch.float16)
    result = process_video(model, config)
    print('Video saved to', result)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <config_file.json>")
        sys.exit(1)
    config_path = sys.argv[1]
    main(config_path)
