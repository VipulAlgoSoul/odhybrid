import yaml
from yaml.loader import SafeLoader

# Open the file and load the file
def load_yaml(yaml_file):
    with open(yaml_file) as f:
        return yaml.load(f, Loader=SafeLoader)
