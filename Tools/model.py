import os
import argparse
import yaml
import subprocess
from log import logger, log

from huggingface_hub import hf_hub_download
import os

model_path = os.path.join(os.getcwd(), "models")


@log()
def download_model(name, repo_id, filenames, output_dir):
    os.makedirs(output_dir, exist_ok=True)  
    if len(filenames) == 1:
        file_path = hf_hub_download(repo_id=repo_id, filename=filenames[0], local_dir=model_path)
        logger.info(f"{name}:{filename} downloaded to: {file_path}")
    else:
        for filename in filenames:
            file_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=model_path)
            logger.info(f"{name}:{filename} downloaded to: {file_path}")

@log()
def main(yaml_path):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    models = config.get('models', [])
    for model in models:
        name = model['name']
        repo_id = model['repo_id']
        filename = model['filename']    # []
        output_dir = model_path
        download_model(name, repo_id, filename, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download model weights from a YAML config.")
    parser.add_argument('--yaml', required=True, help="YAML file describing models to download")
    args = parser.parse_args()

    main(args.yaml)