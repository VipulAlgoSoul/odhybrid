from library.utils import load_yaml
from library.loadmodel import LoadModel


yaml_path = "library/resnet_recipe.yaml"
yaml_cnf = load_yaml(yaml_path)
print(yaml_cnf)

#initiate loading
model_init = LoadModel(yaml_cnf)

#loading base model
model_init.loadbasemodel()

# model_init.eval_on_single_image("E:\odhybrid\\frog.jpg")
# model_init("E:\odhybrid\\frog.jpg")
model_init.test_on_single_image("E:\odhybrid\\frog.jpg")
