import os
import subprocess
import yaml
import argparse
from SpeakerRecognition.log import log, logger

@log()
def run_cmd(cmd_list, cwd=None):
    """Run a command in the shell and return the output."""
    logger.info(f"Running command: {' '.join(cmd_list)}")
    result = subprocess.run(cmd_list, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Command failed with error code {result.returncode}")
    return result.stdout.strip()

@log()
def clone_or_update(repo_url, repo_dir):
    """Clone a git repository and update it."""
    if not os.path.exists(repo_dir):
        logger.info(f"Cloning {repo_url} into {repo_dir}")
        run_cmd(["git", "clone", repo_url, repo_dir], cwd=".")
    else:
        logger.info(f"Updating {repo_dir}")
        run_cmd(["git", "-C", repo_dir, "pull"], cwd=".")

@log()
def build_library(name, build_type, lib_dir):
    logger.info(f"[{name}] Starting build type: {build_type}")
    
    if build_type == "cmake":
        build_path = os.path.join(lib_dir, 'build')
        os.makedirs(build_path, exist_ok=True)
        run_cmd(['cmake', '..'], cwd=build_path)
        run_cmd(['cmake', '--build', '.', '--config', 'Release'], cwd=build_path)

    elif build_type == "autotools":
        autogen = os.path.join(lib_dir, 'autogen.sh')
        if os.path.isfile(autogen):
            run_cmd(['chmod', '+x', 'autogen.sh'], cwd=lib_dir)
            run_cmd(['./autogen.sh'], cwd=lib_dir)
        run_cmd(['./configure'], cwd=lib_dir)
        run_cmd(['make'], cwd=lib_dir)
        run_cmd(['make', 'install'], cwd=lib_dir)

    elif build_type == "none":
        logger.info(f"[{name}] No build step specified. Skipping.")

    else:
        logger.error(f"Unsupported build type: {build_type}")

@log()
def main(yaml_path, output_dir):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    libs = config.get('libraries', [])
    os.makedirs(output_dir, exist_ok=True)

    for lib in libs:
        name = lib['name']
        repo = lib['repo']
        build_type = lib.get('build', 'cmake')
        lib_dir = os.path.join(output_dir, name)

        clone_or_update(repo, lib_dir)
        build_library(name, build_type, lib_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download and build third-party C++ libraries from YAML.")
    parser.add_argument('--yaml', required=True, help='YAML file describing third-party libraries')
    parser.add_argument('--output', default='Packages', help='Directory to clone and build libraries into')
    args = parser.parse_args()

    main(args.yaml, args.output)