from library.utils import load_yaml
from library.loadmodel import LoadModel
import torch
import os

yaml_path = "library/resnet_recipe.yaml"
yaml_cnf = load_yaml(yaml_path)
print(yaml_cnf)

#initiate loading
model_init = LoadModel(yaml_cnf)

#loading base model ---------------------------------
# torch_model = model_init.loadbasemodel()
torch_model = model_init.model_create()
# single image test--------------------------------------
# model_init.test_on_single_image("E:\odhybrid\\frog.jpg")

############################################################
# Write the training loop
# Reshape the model output to the dataloader output

#-----------------------------------------
#Convert to onnx model
torch_input = torch.randn(1, 3, 300, 300) #input image shape
temp_net_path="visualize\\tempnet.onnx"
onnx_program = torch.onnx.export(torch_model, torch_input,temp_net_path)
os.system("netron visualize\\tempnet.onnx")

#-------------------------------------------