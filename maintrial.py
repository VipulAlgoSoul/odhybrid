import ast
import os

import torch
from torch import nn
from torchview import draw_graph

from torchviz import make_dot
from graphviz import Source
#curr_Dir_path nameing has to be changed, they are called in wo places TOC
# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from data_preperation.pipeline import Pipeline
from data_preperation.utils import pathexist
from configs.getconfig import GetConfig
from dataloader.dataloader import CustomImageDataset
import dataloader.dataloader
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from visualize.visualize import visualize_data, visualize_asarray
from prediction.prediction import prediction
from tinkering.tinker import tinkerit

def create_data_dict(data_path, name_fl, np_path):
    '''
    This function creates a dictionary with numpy target key and image path key which have corresponding files
    '''
    ret_dict = {}
    ret_dict["numpy target"]=[]
    ret_dict["image path"]=[]
    fl_path_data = os.path.join(data_path, name_fl)
    im_list = {os.path.splitext(i)[0]: i for i in os.listdir(fl_path_data) if os.path.splitext(i)[-1] != ".txt"}
    for i in os.listdir(np_path):
        np_fl = os.path.join(np_path, i)
        bsnm = os.path.splitext(i)[0]
        # print(bsnm)
        if bsnm in im_list:
            im_nm = im_list[bsnm]
            imp = os.path.join(fl_path_data, im_nm)
            ret_dict["numpy target"].append(np_fl)
            ret_dict["image path"].append(imp)

    return ret_dict


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_cnf_path = "E:\YOLOv10\YOLOv10\configs\data_config.INI"
    init_confg = GetConfig(data_cnf_path)
    init_config = init_confg()

    #read from config
    bypass_pipeline = init_config.getboolean('DATA','bypass_pipeline')
    visualize_traintarget = init_config.getboolean("VISUALIZE","check_target_visualize")

    if not bypass_pipeline:
        set_pipe = Pipeline(data_cnf_path)
        set_pipe()


    ######################################################################################################
    # This section below is getting trainable format data
    ######################################################################################################

    #creat data dict, classes dict, image_shape
    #check for datapath
    #check for np path
    #check for classes text
    exp_path = init_config['PATH']['exp_path']
    data_path = init_config['PATH']['data_folder']
    image_shape = ast.literal_eval(init_config['DATA']['image_shape'])

    curr_data_path = os.path.join(data_path,str(image_shape[0])+"_"+str(image_shape[0]))
    pathexist(curr_data_path)
    np_path = os.path.join(curr_data_path,"npdt")
    sv_dct_units = os.path.join(curr_data_path, str(image_shape[0]) + "dict_cord_units.json")
    sv_asp_ins = os.path.join(curr_data_path, str(image_shape[0]) + "asp_indx_dict.json")

    pathexist(exp_path)
    pathexist(data_path)
    pathexist(np_path)
    pathexist(sv_dct_units)
    pathexist(sv_asp_ins)

    data_dict = create_data_dict(data_path = data_path, name_fl="train", np_path = np_path)
    print(data_dict)

    train_data = CustomImageDataset(data_dict,image_shape, transform=None, target_transform=None)

    train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)
    print("The length of datset is : ",len(train_dataloader.dataset))

    # test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    # if visualize_flag:
    #     visualize_data(train_dataloader)
    #     visualize_asarray(train_dataloader)

    train_features, train_labels = next(iter(train_dataloader))
    # print(f"Feature batch shape: {train_features.size()}")
    # print(f"Labels batch shape: {train_labels.size()}")

    class_path = os.path.join(data_path,"classes.txt")
    #
    if visualize_traintarget:
        pred_icn = prediction(class_path, sv_asp_ins, sv_dct_units, init_config)
        # pred_icn.collect_boxes(train_labels)
        pred_icn.analyse_preds(train_dataloader)

    if torch.cuda.is_available():
        print("Cuda is available")
    else:
        print("cuda not found , going with cpu")

    tink = tinkerit(train_dataloader)
    tink.do()


