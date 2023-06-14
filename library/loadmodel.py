import torch
from torchvision.io import read_image
import torch.nn as nn
from torchsummary import summary


class LoadModel():
    '''This function allows to load base model'''
    def __init__(self, yaml_dict):
        self.yaml_cnf = yaml_dict

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

    def loadbasemodel(self):
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
        print("The number of childrens for basemodel is",len(list(base_model.children())),"**"*20, list(base_model.children()))
        self.base_model = torch.nn.Sequential(*(list(base_model.children())[:-1]))



        if self.yaml_cnf['basemodel_debug']:
            print("The base model is ", self.base_model)
            # print("\nThe model summary is","__"*20)
            # summary(self.base_model)
