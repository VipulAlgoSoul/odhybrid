import torch
from torchvision.io import read_image
import torch.nn as nn
from torchsummary import summary
from configs.getconfig import GetConfig
from .utils import *
import ast
import os

class LoadModel():
    '''This function allows to load base model'''
    def __init__(self, yaml_dict, data_cnf_path="E:\odhybrid\configs\data_config.INI"):
        num_classes =2
        self.yaml_cnf = yaml_dict

        init_confg = GetConfig(data_cnf_path)
        init_config = init_confg()


        self.image_shape = ast.literal_eval(init_config['DATA']['image_shape'])
        self.grids = int(init_config['DATA']['grids'])
        self.per_grid = int(init_config['DATA']['per_grids'])
        self.num_classes = len(read_txt(os.path.join(init_config['PATH']['data_folder'],"classes.txt")))

        # -------------------------------
        # self.grid_jump = int(init_config['DATA']['grid_jump'])
        # self.slider_aspect = init_config.getboolean('DATA', 'slider_aspect')
        # self.recursive_grid_flag = init_config.getboolean('DATA', 'use_recursive_grids')
        # self.recursive_pergrid_flag = init_config.getboolean('DATA', 'use_recursive_pergrids')
        #----------------------------------


        self.output_shape = (1, 2*self.per_grid-1, self.grids*self.grids*(4+self.num_classes+1),1)


    def test_on_single_(self, img, model):

        model.eval()
        # preprocess = self.base_weights.transforms()
        # Step 3: Apply inference preprocessing transforms
        # batch = preprocess(img).unsqueeze(0)
        batch= img.unsqueeze(0)

        # Step 4: Use the model and print the predicted category
        return model(batch).shape

    def test_on_single_image(self, impath):
        img = read_image(impath)


        self.base_model.eval()

        # Step 2: Initialize the inference transforms
        # if self.yaml_cnf['base_transform']:
        preprocess = self.base_weights.transforms()
        # Step 3: Apply inference preprocessing transforms
        batch = preprocess(img).unsqueeze(0)

        # Step 4: Use the model and print the predicted category
        print(self.base_model(batch).shape,">>>>>>>>>>>")
        # prediction = self.base_model(batch)
        # return prediction
    def eval_on_single_image(self, impath):
        img = read_image(impath)


        self.base_model.eval()

        # Step 2: Initialize the inference transforms
        # if self.yaml_cnf['base_transform']:
        preprocess = self.base_weights.transforms()
        # Step 3: Apply inference preprocessing transforms
        batch = preprocess(img).unsqueeze(0)

        # Step 4: Use the model and print the predicted category
        print(self.base_model(batch).shape,">>>>>>>>>>>")
        prediction = self.base_model(batch).squeeze(0).softmax(0)
        class_id = prediction.argmax().item()
        score = prediction[class_id].item()
        category_name = self.base_weights.meta["categories"][class_id]
        print(f"{category_name}: {100 * score:.1f}%")

    def loadbasemodel(self,crop_layer=False):
        # Option 1: passing weights param as string
        if ('basemodel_weights' not in self.yaml_cnf.keys()) or (self.yaml_cnf['basemodel_weights']=='Default'):
            print("Loading {} as base model and initializing {} weights".format(self.yaml_cnf["basemodel"],
                                                                                'Default'))
            self.yaml_cnf['basemodel_weights'] = 'DEFAULT'
        else:
            print("Loading {} as base model and initializing {} weights".format(self.yaml_cnf["basemodel"],
                                                                                self.yaml_cnf['basemodel_weights']))

        # base_model = torch.hub.load("pytorch/vision", self.yaml_cnf["basemodel"], weights=self.yaml_cnf['basemodel_weights'])


        self.base_weights = torch.hub.load("pytorch/vision", "get_weight", self.yaml_cnf['basemodel_weightstring'])
        base_model = torch.hub.load("pytorch/vision", self.yaml_cnf['basemodel'], weights=self.base_weights)

        if not self.yaml_cnf['basemodel_train']:
            for param in base_model.parameters():
                param.requires_grad = False

        # self.base_model.fc=nn.Identity()
        # print(model.classifier)
        del_lyrs = -1*self.yaml_cnf["base_lcnum"]
        print("The number of childrens for basemodel is",len(list(base_model.children())),"**"*20, list(base_model.children()))
        if not crop_layer:
            self.base_model = torch.nn.Sequential(*(list(base_model.children())[:del_lyrs]))
        else:
            self.base_model=base_model

        # if self.yaml_cnf['basemodel_debug']:
        #     print("The base model is ", self.base_model)
        #     # print("\nThe model summary is","__"*20)
        #     # summary(self.base_model)

        return self.base_model

    def model_create(self):
        """This function adds extra layers like FPN, Path aggregations etc,
        The output layer should be in confirmation with output tensor size"""

        #output shape
        # shape: torch.Size([1, 5, 175, 1])
        # print(self.base_model[-1].out_features)
        self.loadbasemodel()
        #create a random input and pass and fetch model outshape
        torch_input = torch.randn(3, self.image_shape[0],self.image_shape[1])  # input image shape
        model_out_shape = self.test_on_single_(torch_input, self.base_model)
        print(model_out_shape,">>>>>>>>>>>>>>>>")

        return self.base_model




