from __future__ import print_function, division
import os
import ast
import shutil

import cv2
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from .utils import save_json_topath

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()

from .create_directory import CreateDirectory
from .pipeline_help import get_data_to_pipe
from configs.getconfig import GetConfig
from .visualize import draw_from_imtxt, get_cords_from_yolo, display_image


class Pipeline(Dataset):

    '''This class contains methods which finally gives out dataloader.
    It also stores samples of train data in new dir crated'''

    def __init__(self, data_cnf_path):
        '''Initializes the pipeline class, create exp path dictionary and data path dict'''
        init_confg = GetConfig(data_cnf_path)
        init_config = init_confg()

        self.exp_path = init_config['PATH']['exp_path']
        self.data_path = init_config['PATH']['data_folder']
        self.image_shape = ast.literal_eval(init_config['DATA']['image_shape'])
        self.grids = int(init_config['DATA']['grids'])
        self.per_grid = int(init_config['DATA']['per_grids'])
        self.grid_jump = int(init_config['DATA']['grid_jump'])
        self.iou_score = float(init_config['DATA']['IOU_score'])
        self.area_encap = float(init_config['DATA']['area_encap'])

        self.backround_flag = init_config.getboolean('DATA', 'background')
        self.iou_allflag = init_config.getboolean('DATA', 'IOU_all')
        self.slider_aspect = init_config.getboolean('DATA','slider_aspect')
        self.recursive_grid_flag = init_config.getboolean('DATA','use_recursive_grids')
        self.recursive_pergrid_flag = init_config.getboolean('DATA','use_recursive_pergrids')
        self.boxnm2 = init_config.getboolean('DATA','use_boxnm2')
        self.nptopath = init_config.getboolean('DATA','save_np_to_path')
        self.readCookeddata = init_config.getboolean('DATA','read_from_image_np')
        self.debug_pipeline = init_config.getboolean('DEBUG','pipeline')
        self.debug_pipeline_help = init_config.getboolean('DEBUG','pipeline_help')
        self.visualize_inp = init_config.getboolean('VISUALIZE','input_visualize')

        self.exp_path_dict = CreateDirectory(self.exp_path)
        self.data_path_dict = self.data_path_seperate(self.data_path)


    def __call__(self):

        if self.nptopath:
            curr_dir_nm = str(self.image_shape[0]) + "_" + str(self.image_shape[1])
            self.curr_data_path = os.path.join(self.data_path, curr_dir_nm)
            if os.path.exists(self.curr_data_path):
                shutil.rmtree(self.curr_data_path)
                os.mkdir(self.curr_data_path)
            else:
                os.mkdir(self.curr_data_path)
            np_nm_shp = "npdt"
            sv_np_path = os.path.join(self.curr_data_path, np_nm_shp)
            os.mkdir(sv_np_path)


        self.gip = get_data_to_pipe(grids=self.grids, per_grids=self.per_grid, imshape=self.image_shape,
                                    train_path=self.data_path_dict["train_path"], grid_jump = self.grid_jump,
                                    slider_aspect= self.slider_aspect, recursive_grids=self.recursive_grid_flag,
                                    recursive_pergrids= self.recursive_pergrid_flag,
                                    save_path=self.exp_path_dict["results"],bgcflag= self.backround_flag, debug=self.debug_pipeline_help)

        self.dict_cord_units, self.config_save_dict, self.data_save_dict  = self.gip.do()

        self.asp_indx_dict = {k:i for i,k in enumerate(self.dict_cord_units.keys())}

        if self.nptopath:

            sv_dct_units = os.path.join(self.curr_data_path, str(self.image_shape[0])+"dict_cord_units.json")
            sv_asp_ins = os.path.join(self.curr_data_path, str(self.image_shape[0])+"asp_indx_dict.json")
            json_cu_dc = {str(k): i for k, i in self.dict_cord_units.items()}
            json_asp_dc = {str(k): i for k, i in self.asp_indx_dict.items()}
            save_json_topath(json_cu_dc, sv_dct_units)
            save_json_topath(json_asp_dc, sv_asp_ins)



        # For visualizing input
        # if self.visualize_inp:
        #     self.input_visualize()

        mapping_df ={"image path":[], 'target layer':[]}
        for imp, im_an in zip(self.data_save_dict["Image Path"], self.data_save_dict["Annotation pipeline"]):
            im_ann = ast.literal_eval(im_an)
            img = cv2.imread(imp).copy()
            img = cv2.resize(img, self.image_shape)
            # print(im_ann)

            # single_cord_dict =self.collect_single_old(im_ann)
            single_cord_connects = self.collect_gtconnects(im_ann, self.iou_score, img)
            # The collect_gtcordsconnects contains key value pairs of each annotation in an image
            # {key : { gt_annot: [ original annotation:[....],"connections":{cnf1:[aspct, grid.....], cnfn:[]}}}


            # display to be uncommented
            if self.visualize_inp:
                self.display_singledict(single_cord_connects, img, "Orig and optimized")

            #create individual targets
            target_array = self.create_trainable(dcu_self=self.dict_cord_units,
                                                 scc =single_cord_connects,one_hot=True)

            mapping_df['image path'].append(imp)
            mapping_df['target layer'].append(target_array)

            if self.nptopath:
                # save asp_indx_dict,dict_cord_units as json
                np_fl = os.path.basename(imp).split('.')[0]+".npy"
                sv_numpy_file = os.path.join(sv_np_path, np_fl)
                np.save(sv_numpy_file, target_array)

        sv_cnf_dict = os.path.join(self.curr_data_path, str(self.image_shape[0]) + "data_config.json")
        save_json_topath(self.config_save_dict, sv_cnf_dict)


    def create_trainable(self, dcu_self, scc,one_hot=True):
        '''
        This method is responsible for craeting output format
        dcu : dict_cord_units, which has keys as aspects and list of bbox values, bbox fromat is [*, xc,yc,w,h]
        scc :single_cord connect, which contains gt and connects of object annotations
        num_classes : is the total number of classes we are training out model for
        one_hot is to use one hot encoding for class representations
        image_p: path to image
        '''

        class_dict = self.config_save_dict['class map']
        num_classes = len(class_dict)
        dcu = {i:tuple(v) for i,v in dcu_self.items()}
        #cord index
        num_op_chnls = len(dcu)

        if "bg" in class_dict.values():
            rop =(5+num_classes+1)*len(list(dcu.values())[0])
        else:
            rop = (5+num_classes)*len(list(dcu.values())[0])


        # print(rop, "the number of rows ")
        op_t = np.zeros((num_op_chnls,rop,1)) #channel row, col
        self.config_save_dict['Target Shape'] = op_t.shape
        # print(op_t.shape, "target vector shape")

        #iterate through scc , enum
        #1 collect class, convert to onehot
        # {0: {'gt_annot': [0.0, 43, 183, 80, 111], 'connections': {0.5754654983570646: [((3, 3), [0.0, 37.5, 187.5, 75.0, 75.0])]},
        for enum, enm_val in scc.items():
            enum_cl = enm_val['gt_annot'][0]
            gt_bx = enm_val['gt_annot'][1::]
            onht = self.create_onehot(num_classes, enum_cl)
            # print("The one hot rep is ", onht, enum_cl)
            enum_cncts = enm_val['connections']
            for cnf, bbxlist in enum_cncts.items():
                for bbs in bbxlist:
                    asp, bx = bbs #here box is [*,x,y,w,h]
                    box=bx[1::]
                    opchannel_index = self.asp_indx_dict[asp]
                    row_indx = dcu[asp].index(box)
                    # print("the row index is ",row_indx)
                    # box_diffn = list(np.array(box.copy())-np.array(gt_bx.copy()))
                    # print("the box diff n", box_diffn)
                    if cnf <= self.iou_score:
                        onht = [0.0]*num_classes

                    use_boxnm2=self.boxnm2
                    # 000
                    if cnf >= self.iou_score:
                        if use_boxnm2:
                            box_diffs = self.box_diff_normalize2(box, gt_bx)
                        else:
                            box_diffs = self.box_diff_normalize(box, gt_bx)
                    else:
                        if use_boxnm2:
                            box_diffs = self.box_diff_normalize2(box, box)

                        else:
                            box_diffs = self.box_diff_normalize(box, box)
                    #x,y,dx,dy,

                    #objectness score asume iou
                    obj_score = cnf

                    # print("__"*40,"\n")
                    # print(onht,obj_score ,box_diffs)

                    sin_tar = onht.copy()
                    sin_tar.append(obj_score)
                    sin_tar.extend(box_diffs)
                    # print(sin_tar)
                    infect = np.transpose(np.array(sin_tar))
                    # print("the channel index is : {} and row index is {}  and len of "
                    #       "vector is {}".format(opchannel_index, row_indx, len(sin_tar)))
                    # print(opchannel_index, op_t.shape, row_indx, row_indx+len(sin_tar), infect.shape)
                    roe_indx = row_indx*len(sin_tar)
                    op_t[opchannel_index, roe_indx:roe_indx+len(sin_tar),0] = infect
                    # print(infect, "The infect")
                    # print(op_t[opchannel_index, row_indx:row_indx + len(sin_tar), 0])

        # print(op_t, "the opt is lkkill")
        return op_t

    def box_diff_normalize2(self, box_an, gt_bx):
        # print(box_an, gt_bx, "breah")
        nw_list = []
        i1 = (box_an[0] - gt_bx[0]) / self.image_shape[1]
        i2 = (box_an[1] - gt_bx[1]) / self.image_shape[0]

        i3 = gt_bx[2] / box_an[2]
        i4 = gt_bx[3] / box_an[3]

        return [i1,i2,i3,i4]



    def box_diff_normalize(self, box, gt_bx):
        box_dif = list(np.array(box.copy()) - np.array(gt_bx.copy()))
        nw_list = []
        for i,val in enumerate(box_dif):
            if i in [0,2]:
                nw = val/self.image_shape[1]
            else:
                nw = val/self.image_shape[0]
            nw_list.append(nw)
        return nw_list
    def create_onehot(self,n,i):
        c = [0.0]*n
        c[int(i)]=1
        return c

    def display_singledict(self,gt_all_dict, img, title ="window"):

        '''

        This class method displays Oringinal bounding box and corresponding IOU anchor
        box in an image.

        :param single_cord_dict: IOU anchor box collected from collect_single class method.
        :param img: corresponding input image.
        :param title: name of the image window.
        :return: None

        '''

        for num, ann_p in gt_all_dict.items():
            imc = img.copy()
            gt_p = get_cords_from_yolo(1, 1, ann_p['gt_annot'])  # class, cordinates

            ann_conval2 = [i for i in ann_p['connections'].values()]
            ann_cnv3 = []
            for i in ann_conval2:
                ann_cnv3.extend(i)

            ls_connections = [get_cords_from_yolo(1, 1, i[-1]) for i in ann_cnv3]
            # ls_connections =[get_cords_from_yolo(1,1,i[-1]) for i in ann_conval]
            imc = cv2.rectangle(imc, gt_p[1], gt_p[2], (0, 255, 0), 3)
            for i in ls_connections:
                imc = cv2.rectangle(imc, i[1], i[2], (255, 0, 0), 3)
            cv2.imshow(title, imc)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    def input_visualize(self):

        '''

        This class method displays images and thier class bounding boxes.

        :return: None
        This class method is just to visualize the input and the class bounding boxes.

        '''

        vis_cnt = 3 # this count is th enumber of image displayed
        image_files = self.data_save_dict["Image Path"]
        text_files = self.data_save_dict["Text Path"]
        anot_fl = self.data_save_dict["Annotation pipeline"]
        class_maps = self.config_save_dict['class map']

        for img, txt in zip(image_files[0:vis_cnt], text_files[0:vis_cnt]):
            draw_from_imtxt(img, txt, title="Train Samples")

        for img, ann in zip(image_files[0:vis_cnt], anot_fl[0:vis_cnt]):
            im = cv2.imread(img).copy()
            im = cv2.resize(im, self.image_shape)
            for an in ast.literal_eval(ann):
                crds = get_cords_from_yolo(1,1,an)
                class_name = class_maps[crds[0]]
                cv2.rectangle(im, crds[1], crds[2], (0, 0, 0) , 3)
                cv2.putText(im, class_name, crds[1],cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.6, color = (255, 0, 0), thickness=2)
            display_image(im,title="Annotation for train" )



    def data_path_seperate(self, data_path):

        '''

        This class method gives out path dictionary.

        :param data_path: input parent folder path where the train, test and validate data is to be looked for.
        :return: path dictionary with train_path, test_path and validate_path.

        '''

        if not os.path.exists(data_path):
            raise Exception("The path to data doesnot exist")
        data_path_dict ={}
        data_path_dict["train_path"] = os.path.join(data_path, "train")
        data_path_dict["test_path"] = os.path.join(data_path,"test")
        data_path_dict["validate_path"] = os.path.join(data_path,"validate")

        return data_path_dict

    def collect_single_old(self, box_crds):

        '''

        This class method returns a dictionary of annotated bounding box as key and its
        corresponding highest IOU anchor box as value.

        :param box_crds: The coordinates of class bounding boxes, output from pipeline_help.
        :return: IOU based anchor coordinates.

        '''

        crd_dict ={}
        for crds in box_crds:
            #this is going through each image annotations
            ann_cord = get_cords_from_yolo(1,1, crds)
            crd_temp_dict ={}

            for key, val in self.dict_cord_units.items():
                key_chann_dict ={}
                for grid_crdes in val:
                    grid_crds =[ann_cord[0]]
                    grid_crds.extend(grid_crdes)
                    anchr_crds = get_cords_from_yolo(1,1,grid_crds)
                    iou_val = self.IOU_cords(ann_cord, anchr_crds)
                    key_chann_dict[iou_val] = grid_crds
                if self.debug_pipeline:
                    print("Cord Annotate, Grid channel \n",key_chann_dict)
                cnf_hier = sorted(list(key_chann_dict.keys()).copy(), reverse=True)
                max_cnf = max(list(key_chann_dict.keys()))
                max_crd = key_chann_dict[max_cnf]
                crd_temp_dict[max_cnf] = (key, max_crd) #from here we can collect the max
            if self.debug_pipeline:
                print("The final channel and corddict \n", crd_temp_dict)
            maxx_select = max(list(crd_temp_dict.keys()))
            crd_dict[str(crds)] = [maxx_select, crd_temp_dict[maxx_select][0], crd_temp_dict[maxx_select][1]]
        if self.debug_pipeline:
            print("Final cord text to channel out \n", crd_dict)

        return crd_dict


    def collect_gtconnects(self, box_crds, iou_score, img):
        '''

        This class method returns a dictionary of annotated bounding box as key and its
        corresponding IOU anchor box as value.

        :param box_crds: The coordinates of class bounding boxes, output from pipeline_help.
        :return: IOU based anchor coordinates.

        '''

        gt_all_dict ={}

        for en_num,crds in enumerate(box_crds):
            #this is going through each image annotations
            gt_all_dict[en_num]={}
            gt_all_dict[en_num]['gt_annot'] = crds

            ann_cord = get_cords_from_yolo(1,1, crds)
            imn = img.copy()
            imn = cv2.rectangle(imn, ann_cord[1], ann_cord[2], (0, 255, 0), 3)

            # collect for an gt ann
            # key_asp_dict={}
            key_chann_dict = {}
            for key, val in self.dict_cord_units.items():
                # key is the aspect, val is list of bboxes of the aspects
                imnn= imn.copy()
                for grid_crdes in val:
                    grid_crds =[ann_cord[0]]
                    grid_crds.extend(grid_crdes)
                    anchr_crds = get_cords_from_yolo(1,1,grid_crds)
                    # iou_val= self.IOU_cords2(ann_cord, anchr_crds)
                    iou_val, encap_score = self.IOUENCAP_cords(ann_cord, anchr_crds)


                    if encap_score>self.area_encap:
                        if iou_val not in key_chann_dict.keys():
                            key_chann_dict[iou_val] =[]
                            key_chann_dict[iou_val].append((key, grid_crds))
                        else:
                            key_chann_dict[iou_val].append((key,grid_crds))

            gt_all_dict[en_num]['connections_lookup'] = key_chann_dict
        # 10 / 0

        # vislz = True
        # self.show_gt_connect(gt_all_dict, img)



        optimized_iou = self.IOU_optimize(img,gt_all_dict=gt_all_dict, iou_score=iou_score)
        for i in optimized_iou.keys():
            gt_all_dict[i]['connections'] = optimized_iou[i]['connections']


        # to visualize
        # if self.visualize_inp:
        vislz =False
        if vislz:
            self.display_singledict(gt_all_dict,img, "post optimization")

        return gt_all_dict

    def IOU_optimize(self,img, gt_all_dict, iou_score):
        '''
        This function optimize the all dictionary
        '''

        # iterate through each gt num
        # collect each connections and their iou values
        aspbx_keys = {}
        # self.show_gt_connect(gt_all_dict,img, "Inside IOU")
        #here is where i have to change
        for enum , ind_dict in gt_all_dict.items():
            connections_lkp = ind_dict['connections_lookup']
            #each connections lookup contains iou score as key and and value of list of tuples
            for iou_k, val_lis in connections_lkp.items():
                for aspcts in val_lis:
                    asp = "#".join([str(i) for i in aspcts[1][1::]])
                    if asp not in aspbx_keys.keys():
                        aspbx_keys[asp] = [iou_k, enum]
                    else:
                        prv_iou = aspbx_keys[asp][0]
                        if prv_iou<iou_k:
                            aspbx_keys[asp] = [iou_k, enum]

        # Find box belongings from aspbx_key,enum
        belong_dict ={}
        for asp, iou_enum in aspbx_keys.items():
            enum_key = iou_enum[-1]
            if enum_key not in belong_dict.keys():
                belong_dict[enum_key]=[]
                belong_dict[enum_key].append(asp)
            else:
                belong_dict[enum_key].append(asp)

        # print("the belong dict", belong_dict)
        vis_beldict=False
        if vis_beldict:
            print("visualizing belong dict")
            self.visualize_belongdict(belong_dict, img, gt_all_dict)
        nw_dict ={}
        for enum , ind_dict in gt_all_dict.items():
            nw_dict[enum]={}
            connections_lkp = ind_dict['connections_lookup']
            app_dict = {}
            nw_dict[enum]['connections']={}

            #each connections lookup contains iou score as key and and value of list of tuples
            for iou_k, val_lis in connections_lkp.items():

                for aspcts in val_lis:
                    asp = "#".join([str(i) for i in aspcts[1][1::]])
                    if asp in belong_dict[enum]:
                        if iou_k not in app_dict.keys():
                            app_dict[iou_k] = []
                            app_dict[iou_k].append(aspcts)
                        else:
                            app_dict[iou_k].append(aspcts)

            # app_dict_app = {i:v for i,v in app_dict.items() if i>=iou_score}
            app_dict_app = {i: v for i, v in app_dict.items()}

            if not bool(app_dict_app):
                # print("the appidct app is", app_dict_app)
                cnf_hier = sorted(list(app_dict.keys()).copy(), reverse=True)
                app_dict_app = {i: app_dict[i] for i in cnf_hier[0:1]}

            nw_dict[enum]['connections'] = app_dict_app

        return nw_dict

    def visualize_belongdict(self, beldict, image, gt_all_dict):
        for enum, ls in beldict.items():
            fdict= gt_all_dict[enum]
            gt_crd = fdict["gt_annot"]
            gt_crd = get_cords_from_yolo(1,1,gt_crd)
            pp = image.copy()
            pp=cv2.rectangle(pp, gt_crd[1], gt_crd[2],(255,0,0),4)
            for i in ls:
                #get values asp = "#".join([str(i) for i in aspcts[1][1::]])
                vals = [float(v) for v in i.split("#")]
                nn =[0]
                nn.extend(vals)
                crd = get_cords_from_yolo(1,1,nn)

                pp = cv2.rectangle(pp, crd[1], crd[2],(0,255,255),3)
            cv2.imshow("beldict", pp)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def ENCAP_cords(self, annot_crd, anchor_crd):

        '''

        This class methods calculates the IOU between two coordinates.

        :param annot_crd: Cordinate of single class bounding box.
        :param anchor_crd: Archor cordinate currently looked into by algorithm.
        :return: IOU score.

        '''

        annot_x_set = set(np.arange(annot_crd[1][0], annot_crd[2][0]))
        annot_y_set = set(np.arange(annot_crd[1][1], annot_crd[2][1]))

        anchor_x_set = set(np.arange(anchor_crd[1][0], anchor_crd[2][0]))
        anchor_y_set = set(np.arange(anchor_crd[1][1], anchor_crd[2][1]))

        x_inter = annot_x_set.intersection(anchor_x_set)
        y_inter = annot_y_set.intersection(anchor_y_set)

        area_inter = len(x_inter) * len(y_inter)

        x_union = annot_x_set.union(anchor_x_set)
        y_union = annot_y_set.union(anchor_y_set)

        area_anchr = len(anchor_x_set) * len(anchor_y_set)

        area_union = len(x_union) * len(y_union)

        return area_inter / area_anchr

    def IOUENCAP_cords(self, annot_crd, anchor_crd):

        '''

        This class methods calculates the IOU between two coordinates.

        :param annot_crd: Cordinate of single class bounding box.
        :param anchor_crd: Archor cordinate currently looked into by algorithm.
        :return: IOU score.

        '''

        annot_x_set = set(np.arange(annot_crd[1][0], annot_crd[2][0]))
        annot_y_set = set(np.arange(annot_crd[1][1], annot_crd[2][1]))

        anchor_x_set = set(np.arange(anchor_crd[1][0], anchor_crd[2][0]))
        anchor_y_set = set(np.arange(anchor_crd[1][1], anchor_crd[2][1]))

        x_inter = annot_x_set.intersection(anchor_x_set)
        y_inter = annot_y_set.intersection(anchor_y_set)

        area_inter = len(x_inter) * len(y_inter)

        x_union = annot_x_set.union(anchor_x_set)
        y_union = annot_y_set.union(anchor_y_set)

        area_anchr = len(anchor_x_set) * len(anchor_y_set)

        area_union = len(x_union) * len(y_union)

        return area_inter/area_union, area_inter / area_anchr


    def IOU_cords(self, annot_crd, anchor_crd):

        '''

        This class methods calculates the IOU between two coordinates.

        :param annot_crd: Cordinate of single class bounding box.
        :param anchor_crd: Archor cordinate currently looked into by algorithm.
        :return:, IOU score.

        '''

        annot_x_set = set(np.arange(annot_crd[1][0], annot_crd[2][0]))
        annot_y_set = set(np.arange(annot_crd[1][1], annot_crd[2][1]))

        anchor_x_set = set(np.arange(anchor_crd[1][0], anchor_crd[2][0]))
        anchor_y_set = set(np.arange(anchor_crd[1][1], anchor_crd[2][1]))

        x_inter = annot_x_set.intersection(anchor_x_set)
        y_inter = annot_y_set.intersection(anchor_y_set)

        area_inter = len(x_inter)*len(y_inter)

        x_union = annot_x_set.union(anchor_x_set)
        y_union = annot_y_set.union(anchor_y_set)

        area_anchr = len(anchor_x_set)*len(anchor_y_set)

        area_union = len(x_union)*len(y_union)

        return area_inter/area_union, area_inter/area_anchr


    def show_gt_connect(self, gt_all_dict,img, ttl="gt_conn_prior"):

        for num, ann_p in gt_all_dict.items():
            imc = img.copy()
            gt_p = get_cords_from_yolo(1, 1, ann_p['gt_annot'])  # class, cordinates
            # print(ann_p['connections'])
            # ann_conval = [i[-1] for i in ann_p['connections'].values()]
            ann_conval_t = [i for i in ann_p['connections_lookup'].values()]

            ann_convalt2 =[]
            for a in ann_conval_t:
                ann_convalt2.extend(a)
            ann_conval = [i[-1] for i in ann_convalt2]
            ls_connections = [get_cords_from_yolo(1, 1, i) for i in ann_conval]
            imc = cv2.rectangle(imc, gt_p[1], gt_p[2], (0, 255, 0), 3)
            for i in ls_connections:
                imc = cv2.rectangle(imc, i[1], i[2], (255, 0, 0), 3)
            cv2.imshow(ttl, imc)
            cv2.waitKey(0)
            cv2.destroyAllWindows()