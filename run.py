from library.utils import load_yaml
from library.loadmodel import LoadModel
from visualize.visualize import count_parameters, save_model_architecture


yaml_path = "E:\YOLOv10\YOLOv10\library//recipe.yaml"
yaml_cnf = load_yaml(yaml_path)

#initiate loading
model_init = LoadModel(yaml_cnf)

#loading base model
base_model, ck = model_init.loadbasemodel()
count_parameters(base_model)
# count_parameters(ck)
#visualize model

model_init.eval_on_single_image("E:\YOLOv10\YOLOv10//frog.jpg")
# save_model_architecture(base_model,chk_out,"myarch")
