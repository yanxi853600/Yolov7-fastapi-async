#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

from genericpath import exists
import cv2
import numpy as np
import json
import os 
from datetime import datetime


def write_in_json(txt_file, read_txt, today):
    while  read_txt :
        lines  =  read_txt.readlines()
        
        if  not  lines:
            break

        chk_json_exist = os.path.isfile(txt_file)  
        if chk_json_exist == False:
            data0 = {
                'file_name': txt_file,
                'AI_start_time': today,
                'AI_Object_Tracking': [
                    
                ]
            }
            


        for  line  in  lines:
            print (line)
            ## call variable
            frame_id = line.split(',')[0]
            track_cls = line.split(',')[1]
            track_id = line.split(',')[2]
            x1 = line.split(',')[3]
            y1 = line.split(',')[4]
            x2 = line.split(',')[5]
            y2 = line.split(',')[6]
            score_ = line.split(',')[7]
            score = score_.split(' ')[0]

            ## write in json


            

today = datetime.now().strftime('%Y-%m-%d')
txt_file = os.path.join('C:/home/ims/yolov7/Bytetrack_yolov7/result/TBN4223133_1_20220613-155351_4s.txt')
json_file = os.path.join('C:/home/ims/yolov7/Bytetrack_yolov7/result/TBN4223133_1_20220613-155351_4s.json')
read_txt = open(txt_file, 'r', encoding='utf-8') 


write_in_json(txt_file, read_txt, today)






    
    
    # chk json file
    # chk_json_exist = os.path.isfile(res_file)  
    # if chk_json_exist == False:
    #     data0 = {
    #         'file_name': res_file,
    #         'AI_start_time': today,
    #         'AI_Object_Tracking': [
                
    #         ]
    #     }
    # else:
    #     with open(res_file, "r+") as f:
    #         data0 = json.loads(f.read())
    # data1 = {
    #             'frame': frame_id,
    #             'object': [
    #                 {
    #                     'track_cls': t_label,
    #                     'track_id': tid,
    #                     'score': t_score,
    #                     'x1': tlwh_0,
    #                     'y1': tlwh_1,
    #                     'x2': tlwh_2,
    #                     'y2': tlwh_3
    #                 }
    #             ]
    #         }

    # # with open(res_file, 'w', encoding='utf-8') as f:
    # data0['AI_Object_Tracking'].append(data1)

    # str_ = json.dumps(data0,
    #                 indent=4, sort_keys=False,
    #                 separators=(',', ': '), ensure_ascii=False)
    # write_json.write(str_)
