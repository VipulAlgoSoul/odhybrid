from library.utils import load_yaml
from library.loadmodel import LoadModel


yaml_path = "E:\YOLOv10\YOLOv10\library//recipe.yaml"
yaml_cnf = load_yaml(yaml_path)

#initiate loading
model_init = LoadModel(yaml_cnf)

#loading base model
model_init.loadbasemodel()

model_init.eval_on_single_image("E:\YOLOv10\YOLOv10//frog.jpg")
