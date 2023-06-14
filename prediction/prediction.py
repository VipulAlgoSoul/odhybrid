import os
import ast
import json

import torch
import cv2

from data_preperation.utils import read_txt, read_json
from data_preperation.visualize import get_cords_for_pred, display_image, get_cords_from_yolo
from configs.getconfig import GetConfig

import numpy as np

class prediction():
    '''This class is uesd to craete a prediction method'''

    def __init__(self,class_path, asp_index , dcu, init_config):
        """
        class map: mapping of classes
        asp_index: index of aspects
        dict_cord_units: units of dict cords
        """

        class_lines = read_txt(class_path)
        n_class = len(class_lines)
        self.class_map_dict = {i: cls for i, cls in enumerate(class_lines)}
        asp_id = read_json(asp_index)
        self.asp_idx = {int(v):k for k,v in asp_id.items()}
        self.dcu_dict=read_json(dcu)
        self.block_len = n_class+5 #1 for background possible bug in the code

        self.iterator_range = len(list(self.dcu_dict.values())[0])
        self.boxnm2 = init_config.getboolean('DATA','use_boxnm2')
        self.image_shape = ast.literal_eval(init_config['DATA']['image_shape'])

    def collect_boxes(self, target, image):
        img=image.squeeze()
        img= img.permute(1,2,0)
        imgnp = img.numpy()

        tg = target.squeeze(0)
        the_hold_dict={}

        #convert tensor to numpy array
        for tg_asp in range(tg.shape[0]): #iterate through target channels, ie; aspects

            lkt = tg[tg_asp,:,:]
            # print("the lkt", lkt)
            lkt_list = lkt.squeeze(-1)

            feelr = {i:val for i,val in enumerate([lkt_list[m*self.block_len:m*self.block_len+self.block_len]
                                                   for m in range(self.iterator_range)]) if torch.count_nonzero(val[0:-4])>0}
            # the_hold_dict[tg_asp]=feelr
            ifer = self.collect_ni_comparison(feelr,tg_asp)
            # print("Ifer keys of detections >>>>>>", ifer)
            the_hold_dict[tg_asp] = ifer

        self.pred_visualize(the_hold_dict, imgnp)

    def pred_visualize(self, the_hold_dict, image):
        val_list = [i for i in the_hold_dict.values() if i]
        for m in val_list:
            img = image.copy()
            for v in m:
                cnf_scr = v[0].detach().numpy()
                class_obj= v[-1][0].detach().numpy()

                class_name = self.class_map_dict[int(class_obj)]
                class_name = class_name+" : "+str(cnf_scr)[0:4]
                img = cv2.rectangle(img, v[-1][1],v[-1][2], (0,255,0),3)
                img = cv2.putText(
                    img,
                    text=class_name,
                    org=v[-1][1],
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0, 0, 255),
                    thickness=3
                )
            display_image(img,self.image_shape, title="on")

    def collect_ni_comparison(self, feelr,tg_asp):
        new_ls =[]
        the_asp = self.asp_idx[tg_asp]
        for idx, pnts in feelr.items():
            asp_box = self.dcu_dict[the_asp][idx]
            if self.boxnm2:
                pred_box, obj_class, obj_cnf =self.box_diff_normalize2(pnts, asp_box)
            else:
                10/0
                obj_cnf=1 ####TOC
                obj_class=1
                pred_box=1
            yolofrm = [obj_class]
            yolofrm.extend(pred_box)

            # new_ls.append([obj_cnf,get_cords_for_pred(yolofrm, self.image_shape)])
            new_ls.append([obj_cnf, get_cords_from_yolo(1,1,yolofrm)])
        return new_ls

    def box_diff_normalize2(self, pred_an, anc_bx):
        new_pred= []
        i1 = pred_an[-4]
        i2 = pred_an[-3]

        i3 = pred_an[-2]
        i4 = pred_an[-1]
        ####
        new_pred.append(int(anc_bx[0] - i1 * self.image_shape[1]))
        new_pred.append(int(anc_bx[1] - i2 * self.image_shape[0]))

        # delx =
        new_pred.append(int(i3 * anc_bx[2]))
        new_pred.append(int(i4 * anc_bx[3]))

        objscore = pred_an[-5]
        cls_obj = self.block_len-5 ###################################TOC
        class_t = torch.argmax(pred_an[0:cls_obj])

        # if backgroun:
        #     print("dcct would be different")


        return new_pred, class_t, objscore*torch.max(pred_an[0:cls_obj])



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


    def analyse_preds(self,traindata ):

        for i, (images, labels) in enumerate(traindata):
            self.collect_boxes(labels, images)






