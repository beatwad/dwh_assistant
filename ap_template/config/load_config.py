import yaml
import os

config_file = "config/config.yaml"

def load_config() -> dict:
    """Load configuration variables from YAML file"""
    with open(config_file, 'r') as f:
        yaml_dict = yaml.safe_load(f)
    return yaml_dict
