# config_loader.py
from omegaconf import OmegaConf

def load_config(yaml_path):
    return OmegaConf.load(yaml_path)