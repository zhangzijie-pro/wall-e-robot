import yaml
import os

def load_config(package_name, file_name):
    # 查找包路径
    from ament_index_python.packages import get_package_share_directory
    path = os.path.join(get_package_share_directory(package_name), 'config', file_name)
    with open(path, 'r') as f:
        return yaml.safe_load(f)
