import yaml
from yaml.loader import SafeLoader

# Open the file and load the file
def load_yaml(yaml_file):
    with open(yaml_file) as f:
        return yaml.load(f, Loader=SafeLoader)


def read_txt(txt_path):
    '''read text files'''
    with open(txt_path) as f:
        lines = f.readlines()
    return lines
