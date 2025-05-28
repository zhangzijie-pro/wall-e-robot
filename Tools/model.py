import os
import argparse
import yaml
import subprocess
import urllib.request
from log import logger, log

@log()
def run_cmd(cmd_list, cwd=None):
    """Run a command in the shell and return the output."""
    logger.info(f"Running command: {' '.join(cmd_list)}")
    result = subprocess.run(cmd_list, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Command failed with error code {result.returncode}")
    return result.stdout.strip()

@log()
def download_model(name, url, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, os.path.basename(url))

    if os.path.exists(filename):
        logger.info(f"[{name}] Model already downloaded: {filename}")
    else:
        logger.info(f"[{name}] Downloading from {url} to {filename}")
        run_cmd(["wget", url, "-O", filename])

@log()
def main(yaml_path):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    models = config.get('models', [])
    for model in models:
        name = model['name']
        url = model['url']
        output_dir = model.get('output_dir', 'downloaded_models')
        model = os.path.abspath(os.path.join("..",output_dir))
        download_model(name, url, model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download model weights from a YAML config.")
    parser.add_argument('--yaml', required=True, help="YAML file describing models to download")
    args = parser.parse_args()

    main(args.yaml)
