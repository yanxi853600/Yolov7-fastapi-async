import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.plots import plot_one_box
from utils.torch_utils import time_synchronized, TracedModel
import time 
from tracker.byte_tracker import BYTETracker
from utils.visualize import plot_tracking
# from utils.txt_json import write_in_json
from tracking_utils.timer import Timer
from inference import Detect
import sys
import imageio
import json
from datetime import datetime
from collections import deque
import signal
# Add Ipcmaaera lib
from utils.IP_Camera import ipcamCapture
# fastapi
# from fastapi.responses import StreamingResponse
from fastapi import FastAPI, Form, File, UploadFile, Header, Query, Request, Response
# from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import requests
# from starlette.responses import FileResponse
# from starlette.responses import StreamingResponse
# from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware

import uvicorn
import shutil
from requests.exceptions import Timeout
from utils.response import ApiResponse
from utils.exceptions import ModelNotFound, InvalidModelConfiguration, ApplicationError, ModelNotLoaded, \
	InferenceEngineNotFound, InvalidInputData
from utils.errors import Error
import threading
from func_timeout import func_timeout, FunctionTimedOut, func_set_timeout
from collections import Counter
from utils.General_Api import Cal_IOU
from utils.ShareMemory import SharedMemory_Image
from pydantic import BaseModel
from optparse import Option, OptionParser
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

error_logging = Error()
#####################################################
# 	API Release Information (http://127.0.0.1:8888/docs)
#####################################################
app = FastAPI(version="1.0.0", title='Yolov7 inference Swagger',
			  description="<b>API for performing YOLOv7 inference.</b></br></br>"
			 )
#####################################################
#	CORS Setting
#####################################################
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
	max_age=180,	#	timout (second)
)

class Item_prevtime(BaseModel):
    PrevTime:int

class Item_path(BaseModel):
    img_path:str

class Item(BaseModel):
    LabelName:str
    StartupPoint:str
    EndPoint:str
    TimeOut:int
    ContinuousTime:int
    CheckFlag:int
    ObjectCount:int
    IouMode:int
    MeasureBased:int


class Node:
  def __init__(self, data):
    self.data = data
    self.next = None

class L_List:
    def __init__(self):
        self.head = None

    def push_(self, newElement):
        newNode = Node(newElement)
        if(self.head == None):
            self.head = newNode
            return
        else:
            temp = self.head
            while(temp.next != None):
                temp = temp.next
            temp.next = newNode

    def loop_(temp, l_, CMD_ContinuousTime):
        l_.append(temp.data)
        if len(l_) == CMD_ContinuousTime * fps * fps:
            return True
       
    def chk_CMD_ContinuousTime(self, l_, CMD_ContinuousTime):
        temp = self.head

        ## reset the head
        if temp.data == False:
            temp = temp.next

        else:
            res_ = L_List.loop_(temp, l_, CMD_ContinuousTime)
            if res_ == None:
                temp = temp.next
        return temp
            

def parse_args():
    with open('./config/DAI_setting.json', 'r') as in_file:
        text = in_file.read()
        Setting_Option = json.loads(text)
        parser = OptionParser()

        ## AI setting
        parser.add_option('--weights', type=str, default=Setting_Option['AI_Setting']['weights'])
        parser.add_option('--conf_thres', type=float, default=Setting_Option['AI_Setting']['conf_thres'])
        parser.add_option('--cal_iou', type=float, default=Setting_Option['AI_Setting']['cal_iou'])
        parser.add_option('--img_size', type=int, default=Setting_Option['AI_Setting']['img_size'])
        parser.add_option('--device', type=int, default=Setting_Option['AI_Setting']['device'])
        parser.add_option('--track_id', type=str, default=Setting_Option['AI_Setting']['track_id'])
        parser.add_option('--ip', type=str, default=Setting_Option['AI_Setting']['ip'])
        parser.add_option('--port', type=int, default=Setting_Option['AI_Setting']['port'])
        parser.add_option('--cam_ip', type=str, default=Setting_Option['AI_Setting']['cam_ip'])
        
        ## Video setting
        parser.add_option('--show_video', type=str, default=Setting_Option['Video_Setting']['show_video'])
        parser.add_option('--save_video', type=str, default=Setting_Option['Video_Setting']['save_video'])
        parser.add_option('--output_video_path', type=str, default=Setting_Option['AI_Setting']['output_video_path'])

        ## Other setting
        parser.add_option('--video_test', type=str, default=False)
        parser.add_option('--txt_dir', type=str, default='result')
        parser.add_option('--iou_thres', type=float, default=Setting_Option['AI_Setting']['iou_thres'])
        parser.add_option('--track_thresh', type=float, default=0.5)
        parser.add_option('--track_buffer', type=int, default=30)
        parser.add_option('--match_thresh', type=float, default=0.8)
        parser.add_option('--frame_rate', type=int, default=25)
        parser.add_option('--aspect_ratio_thresh', type=float, default=1.6) # -1 means no need to filter out bboxes, usuallly set 1.6 for pedestrian
        parser.add_option('--min_box_area', type=int, default=10) ## 0 means no need to filter out too small boxes
        parser.add_option('--half_precision', type=str, default=True)
        parser.add_option('--writer', type=str, default=None)
        parser.add_option('--mot20_check', type=str, default=False)

        (opt, args) = parser.parse_args()

    return opt

def PrevTime_chk(_prevtime, debug=False):
    try:
        _PT = _prevtime['PrevTime']
    except:
        print('ERROR - Json Format.')
  

    return _PT

def CMD_Check(_CMD, debug=False):

    try:
        # Defect Object information
        _LabelName = _CMD['LabelName']
        _StartupPoint = _CMD['StartupPoint']
        _EndPoint = _CMD['EndPoint']
        _ObjectCount = _CMD['ObjectCount']
        _CheckFlag = _CMD['CheckFlag']

        # Check Rule
        _TimeOut = _CMD['TimeOut']
        _ContinuousTime = _CMD['ContinuousTime']
        _IouMode = _CMD['IouMode']
        _MeasureBased = _CMD['MeasureBased']

        
    except:
        print('ERROR - Json Format.')
  

    return _LabelName, _StartupPoint, _EndPoint, _ObjectCount, _CheckFlag, _TimeOut, _ContinuousTime, _IouMode, _MeasureBased


def Yolov7_ByteTrack(opt, deteted, tracker, output_video_path):    
    if not os.path.exists(opt.txt_dir):
        os.makedirs(opt.txt_dir)
    timer = Timer()
    frame_id = 0
    
    ## store in top fps
    q_fps = deque(maxlen=1)

    try:
        global fps, vid_writer
        vid_writer = None
        fps, w, h = cap.getvideotype()
        vid_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, (w, h))
    except Exception as e:
        error_logging.error(str(e))
        return ApiResponse(result=False, error=str(e)+' > check camera is turn-on or not!')

    global G_queue
    fps = 15
    G_queue = deque(maxlen=int(fps))

    ## SharedMemory
    mm = SharedMemory_Image("ShareMemory_Image")

    while True:
        _,im0 = cap.read_()
        frame_id += 1
        # im0 = cap.Frame

        if _:
            height, width, channel = im0.shape
            t1 = time.time()
            dets = deteted.detecte_Yolov7_ByteTrack(im0)
            try:
                if len(dets) != 0:
                    online_targets = tracker.update(np.array(dets), [height, width], (height, width))
                    online_tlwhs, online_ids, online_scores, online_label = [], [], [], []

                    # dequeue to store the newest 10 frames
                    # q = deque()
                    q = []
                    try:
                        for t in online_targets:
                            tlwh = t.tlwh
                            tid = t.track_id
                            online_tlwhs.append(tlwh)
                            online_ids.append(tid)
                            online_scores.append(t.score)
                            
                            label_name = deteted.names[int(t.label)]
                            online_label.append(label_name)

                            # save result for evaluation
                            q.append(
                                f"{frame_id},{label_name},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f} \n" # ,-1,-1,-1
                            )
                    except Exception as e:
                        error_logging.error(str(e))
                        return ApiResponse(result=False, error=str(e)+' > deQue append inside is error!')

                    t2 = time.time()
                    print(f"FPS:{1 /(t2-t1):.2f}")
                    timer.toc()

                    try:
                        names = deteted.names
                        online_im = plot_tracking(im0, names, online_tlwhs, online_ids, online_label, online_scores, frame_id=frame_id + 1, fps=1. / 1 /(t2-t1))
                    except Exception as e:
                        error_logging.error(str(e))
                        return ApiResponse(result=False, error=str(e)+' > Plot tracking img is error!')
                    try:
                        # cv2.imwrite('testing0915.jpg',online_im)
                        mm.WriteMemoryMapped(online_im)
                        q_fps.append(q)
                        
                    except Exception as e:
                        error_logging.error(str(e))
                        return ApiResponse(result=False, error=str(e)+' > deQue append outside is error!!')

                    if opt.save_video == 'True':
                        vid_writer.write(online_im)                    

                    if opt.show_video == 'True':
                        online_im_h = int(online_im.shape[0] / 2.5)
                        online_im_w= int(online_im.shape[1] / 2.5)
                        online_im = cv2.resize(online_im, (online_im_w, online_im_h), interpolation=cv2.INTER_AREA)

                        cv2.namedWindow('DAI_Inference', 0)
                        cv2.resizeWindow('DAI_Inference', (online_im_w, online_im_h))
                        cv2.moveWindow("DAI_Inference", 0, 0)
                        cv2.imshow("DAI_Inference", online_im)
                        ch = cv2.waitKey(1)  # 1 millisecond

                        if ch == ord("q") : 
                            break
                    
                    G_queue += q_fps
                
                else:
                    try:
                        q, q_fps=[], []
                        q_fps.append(q)
                        
                        mm.WriteMemoryMapped(im0)

                        t3 = time.time()
                        print(f"FPS:{1 /(t3-t1):.2f}")
                        timer.toc()

                        if opt.save_video == 'True':
                            vid_writer.write(im0)                    

                        if opt.show_video == 'True':
                            online_im_h = int(im0.shape[0] / 2.5)
                            online_im_w= int(im0.shape[1] / 2.5)
                            online_im = cv2.resize(im0, (online_im_w, online_im_h), interpolation=cv2.INTER_AREA)

                            cv2.namedWindow('DAI_Inference', 0)
                            cv2.resizeWindow('DAI_Inference', (online_im_w, online_im_h))
                            cv2.moveWindow("DAI_Inference", 0, 0)
                            cv2.imshow("DAI_Inference", im0)
                            ch = cv2.waitKey(1)  # 1 millisecond

                            if ch == ord("q") : 
                                break
                        
                        G_queue += q_fps

                    except Exception as e:
                        error_logging.error(str(e))
                
                time.sleep(0.1)

            except Exception as e:
                error_logging.error(str(e))
                return ApiResponse(result=False, error=str(e)+' > model detection is error!')                    
            
        else:
            try:
                ## ipcam disconnection
                cap.stop()
                if cap.isstop == True:
                    ## initialize G_queue 
                    G_queue = deque([],maxlen=int(fps))
                    vid_writer.release()
                    cv2.destroyAllWindows()
                    time.sleep(0.1)
                    error_logging.error('!!! IPcamera is disconnected')
                    return ApiResponse(result=False, error='!!! IPcamera is disconnected') 
            except Exception as e:
                error_logging.error(str(e))
                return ApiResponse(result=False, error=str(e))
        
    
    vid_writer.release()
    cv2.destroyAllWindows()


def Yolov7(opt, deteted, output_video_path, CMD_previos_time):
    if not os.path.exists(opt.txt_dir):
        os.makedirs(opt.txt_dir)
    timer = Timer()
    frame_id = 0
    
    ## store in top fps
    q_fps = deque(maxlen=1)

    try:
        global fps
        vid_writer = None
        if opt.video_test == True:
            fps = cap.get(cv2.CAP_PROP_FPS)
            w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:
            fps, w, h = cap.getvideotype()
        vid_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h)) # *'DIVX'
    except Exception as e:
        error_logging.error(str(e))
        return ApiResponse(result=False, error=str(e)+' > check camera is turn-on or not!')

    global G_queue
    # fps = 15 * CMD_previos_time
    fps = 1 * CMD_previos_time
    G_queue = deque(maxlen=int(fps))

    ## SharedMemory
    mm = SharedMemory_Image("ShareMemory_Image")

    while True:
        if cap.isstop == False:
            if opt.video_test == True:
                _,im0 = cap.read()
            else:
                _,im0 = cap.read_()
            frame_id += 1
        
            # im0 = cap.Frame

            if _:
                height, width, channel = im0.shape
                plot_img, q = deteted.detecte_Yolov7(im0, frame_id)
                try:
                
                    if len(q) != 0:
                        try:
                            mm.WriteMemoryMapped(plot_img)
                            q_fps.append(q)
                            
                        except Exception as e:
                            error_logging.error(str(e))
                            return ApiResponse(result=False, error=str(e)+' > deQue append outside is error!!')

                        if opt.save_video == 'True':
                            vid_writer.write(plot_img)                    

                        if opt.show_video == 'True':
                            plot_img_h = int(plot_img.shape[0]/2.5)
                            plot_img_w= int(plot_img.shape[1]/2.5)
                            plot_img = cv2.resize(plot_img, (plot_img_w, plot_img_h), interpolation=cv2.INTER_AREA)

                            cv2.namedWindow('DAI_Inference', 0)
                            cv2.resizeWindow('DAI_Inference', (plot_img_w, plot_img_h))
                            cv2.moveWindow("DAI_Inference", 0, 0)
                            cv2.imshow("DAI_Inference", plot_img)
                            ch = cv2.waitKey(1)  # 1 millisecond

                            if ch == ord("q") : 
                                break
                        
                        G_queue += q_fps
                
                    else:
                        try:
                            q, q_fps=[], []
                            q_fps.append(q)
                            
                            mm.WriteMemoryMapped(im0)


                            if opt.save_video == 'True':
                                vid_writer.write(im0)                    

                            if opt.show_video == 'True':
                                plot_img_h = int(im0.shape[0]/2.5 )
                                plot_img_w= int(im0.shape[1]/2.5)
                                plot_img = cv2.resize(im0, (plot_img_w, plot_img_h), interpolation=cv2.INTER_AREA)

                                cv2.namedWindow('DAI_Inference', 0)
                                cv2.resizeWindow('DAI_Inference', (plot_img_w, plot_img_h))
                                cv2.moveWindow("DAI_Inference", 0, 0)
                                cv2.imshow("DAI_Inference", im0)
                                ch = cv2.waitKey(1)  # 1 millisecond

                                if ch == ord("q") : 
                                    break
                            
                            G_queue += q_fps

                        except Exception as e:
                            error_logging.error(str(e))

                    time.sleep(0.01)

                except Exception as e:
                    error_logging.error(str(e))
                    return ApiResponse(result=False, error=str(e)+' > model detection is error!')                    
                
        else:
            try:
                ## ipcam disconnection
                # cap.stop()
                # break
                # if cap.isstop == True:
                ## initialize G_queue 
                G_queue = deque([],maxlen=int(fps))
                vid_writer.release()
                cv2.destroyAllWindows()
                error_logging.error('!!! IPcamera is disconnected')
                time.sleep(1)
                return ApiResponse(result=False, error='!!! IPcamera is disconnected') 
            except Exception as e:
                error_logging.error(str(e))
                return ApiResponse(result=False, error=str(e))
        time.sleep(0.01)

    vid_writer.release()
    cv2.destroyAllWindows()

def Yolov7_video(opt, deteted, input_video_path, output_video_path):
    if not os.path.exists(opt.txt_dir):
        os.makedirs(opt.txt_dir)
    timer = Timer()
    frame_id = 0
    
    ## store in top fps
    q_fps = deque(maxlen=1)
    cap = cv2.VideoCapture(input_video_path)

    try:
        global fps
        vid_writer = None
        # fps, w, h = cap.getvideotype()
        # vid_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, (w, h))

        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (w, h))


    except Exception as e:
        error_logging.error(str(e))
        return ApiResponse(result=False, error=str(e)+' > check camera is turn-on or not!')

    global G_queue
    fps = 15
    G_queue = deque(maxlen=int(fps))

    ## SharedMemory
    mm = SharedMemory_Image("ShareMemory_Image")

    while True:
        _,im0 = cap.read()
        frame_id += 1
        # im0 = cap.Frame

        if _:
            height, width, channel = im0.shape
            plot_img, q = deteted.detecte_Yolov7(im0, frame_id)
            try:
                if len(q) != 0:
                    try:
                        mm.WriteMemoryMapped(plot_img)
                        q_fps.append(q)
                        
                    except Exception as e:
                        error_logging.error(str(e))
                        return ApiResponse(result=False, error=str(e)+' > deQue append outside is error!!')

                    if opt.save_video == True:
                        vid_writer.write(plot_img)                    

                    if opt.show_video == 'True':
                        plot_img_h = int(plot_img.shape[0]/2.5)
                        plot_img_w= int(plot_img.shape[1]/2.5)
                        plot_img = cv2.resize(plot_img, (plot_img_w, plot_img_h), interpolation=cv2.INTER_AREA)

                        cv2.namedWindow('DAI_Inference', 0)
                        cv2.resizeWindow('DAI_Inference', (plot_img_w, plot_img_h))
                        cv2.moveWindow("DAI_Inference", 0, 0)
                        cv2.imshow("DAI_Inference", plot_img)
                        cv2.imwrite("DAI_Inference_nms_FTT.jpg", plot_img)
                        ch = cv2.waitKey(1)  # 1 millisecond

                        if ch == ord("q") : 
                            break
                    
                    G_queue += q_fps
            
                else:
                    try:
                        q, q_fps=[], []
                        q_fps.append(q)
                        
                        mm.WriteMemoryMapped(im0)


                        if opt.save_video == 'True':
                            vid_writer.write(im0)                    

                        if opt.show_video == 'True':
                            plot_img_h = int(im0.shape[0]/2.5 )
                            plot_img_w= int(im0.shape[1]/2.5)
                            plot_img = cv2.resize(im0, (plot_img_w, plot_img_h), interpolation=cv2.INTER_AREA)

                            cv2.namedWindow('DAI_Inference', 0)
                            cv2.resizeWindow('DAI_Inference', (plot_img_w, plot_img_h))
                            cv2.moveWindow("DAI_Inference", 0, 0)
                            cv2.imshow("DAI_Inference", im0)
                            ch = cv2.waitKey(1)  # 1 millisecond

                            if ch == ord("q") : 
                                break
                        
                        G_queue += q_fps

                    except Exception as e:
                        error_logging.error(str(e))

                time.sleep(0.1)

            except Exception as e:
                error_logging.error(str(e))
                return ApiResponse(result=False, error=str(e)+' > model detection is error!')                    
            
        else:
            try:
                ## ipcam disconnection
                cap.stop()
                if cap.isstop == True:
                    ## initialize G_queue 
                    G_queue = deque([],maxlen=int(fps))
                    vid_writer.release()
                    cv2.destroyAllWindows()
                    error_logging.error('!!! IPcamera is disconnected')
                    return ApiResponse(result=False, error='!!! IPcamera is disconnected') 
            except Exception as e:
                error_logging.error(str(e))
                return ApiResponse(result=False, error=str(e))
            time.sleep(0.1)
    
    vid_writer.release()
    cv2.destroyAllWindows()


def AI_Object_chk(CMD_LabelName, CMD_StartupPoint, CMD_EndPoint, CMD_ObjectCount, CMD_CheckFlag, CMD_ContinuousTime, CMD_IouMode):
    ## setting
    AI_obj_res = False
    A_obj, B_obj = None, None
    res_iou_list_, CMD_workstation_pos, obj_count_, output_, obj_label_=[], [], [], [], []
    same_id  = {}
    
    CMD_StartupPoint = CMD_StartupPoint.split(',') 
    CMD_EndPoint = CMD_EndPoint.split(',') 
    CMD_workstation_pos.extend([int(CMD_StartupPoint[0]), int(CMD_StartupPoint[1]), int(CMD_EndPoint[0]), int(CMD_EndPoint[1])])
    

    while AI_obj_res == False:
        try:
            time.sleep(0.01)
            ## Check the same obj as cmd
            # base case: end if AI_obj_res
            label_list = deque(maxlen=int(fps))
            AI_list = list(G_queue)

            try:
                for i in AI_list:            
                    target_obj = str(CMD_LabelName) + ','
                    _ = list(filter(lambda x: target_obj in x, i))
                    label_list.append(_)
            except Exception as e:
                print(e)

                
            try:
                ## Parse cmd & calculate IOU    
                res_iou_ = False
                if opt.track_id == "True":
                    for obj in label_list:
                        res_iou_list = []
                        for l in obj:
                            AI_obj_pos = []
                            output=l.split(',')            
                            frame_ = output[0]
                            label_ = output[1]
                            id_ = output[2]
                            label_id_ = output[1]+"_"+output[2]
                            x1_ = int(float(output[3]))
                            y1_ = int(float(output[4]))
                            x2_ = int(x1_) + int(float(output[5]))
                            y2_ = int(y1_) + int(float(output[6]))
                            score_ = output[7]
                            output[1] = label_id_
                            output.remove(id_)
                            obj_label_.append(label_id_)
                            output_+=output

                            ## Calculate IOU
                            if CMD_IouMode == 0:
                                A_obj = CMD_workstation_pos
                                B_obj = AI_obj_pos

                            AI_obj_pos.extend([x1_, y1_, x2_, y2_])
                            res_iou = Cal_IOU.bb_overlab(A_obj, B_obj)
                            if res_iou > opt.cal_iou:
                                res_iou_ =  True
                            else:
                                res_iou_ = False
                            res_iou_list.append(res_iou_)
                        res_iou_list_.append(res_iou_list)
                else:
                    for obj in label_list:
                        res_iou_list = []
                        for l in obj:
                            AI_obj_pos = []
                            output=l.split(',')            
                            
                            x1_ = int(float(output[2]))
                            y1_ = int(float(output[3]))
                            x2_ = int(output[4])
                            y2_ = int(output[5])
                            
                            output_+=output

                            ## Calculate IOU
                            if CMD_IouMode == 0:
                                A_obj = CMD_workstation_pos
                                B_obj = AI_obj_pos

                            AI_obj_pos.extend([x1_, y1_, x2_, y2_])
                            res_iou = Cal_IOU.bb_overlab(A_obj, B_obj)
                            if res_iou > opt.cal_iou:
                                res_iou_ =  True
                            else:
                                res_iou_ = False
                            res_iou_list.append(res_iou_)
                        res_iou_list_.append(res_iou_list)
                                
                ## Calculate diff frames have the same id or not
                try:
                    ori_id, ai_id = Counter(same_id), Counter(output_)
                    sum_ = dict(ori_id+ai_id)
                    same_id = sum_
                    same_id_ = same_id.copy()
                    
                    ## Reset same_id
                    for key, value in same_id.items():
                        if CMD_LabelName not in key:
                            same_id_.pop(key)
                            continue

                except Exception as e:
                        print(e)  
            
                # Check obj has the same id
                for o_ in set(obj_label_):    
                    if same_id_.get(o_) < int(len(label_list)) :
                        print("It's not the same id : It may represents diff obj! ")
                        break


                ## Check Iou
                global record_
                global t_false
                record_ = deque(maxlen=2)
                t_false = 0

                if len(res_iou_list_) < int(fps):
                    continue
                
                else:
                    for res_iou in res_iou_list_:
                        # print(res_iou.count(True))
                        if CMD_CheckFlag == 1:
                            if res_iou.count(True)>= CMD_ObjectCount:
                                AI_obj_res = True
                            else:
                                AI_obj_res = False
                                t_false = time.time()
                                # print('< CMD_ObjectCount : ' + str(time.time()-t_false))

                            record_.append(AI_obj_res)

                        else:
                            if res_iou.count(True) < CMD_ObjectCount:
                                AI_obj_res = True
                            else:
                                AI_obj_res = False
                                t_false = time.time()

                            record_.append(AI_obj_res)
            
            except Exception as e:
                error_logging.error(str(e))
            
        except Exception as e:
            error_logging.error(str(e))
            return ApiResponse(result=False, error=e) 
            
    return AI_obj_res, label_list, res_iou_list_, res_iou.count(True)


opt = parse_args()
print(opt)

## model donwnload
deteted = Detect(opt.weights, opt.device, opt.img_size, opt.conf_thres, opt.iou_thres, single_cls=False, half_precision=opt.half_precision, trace= False)
tracker = BYTETracker(opt.track_thresh, opt.track_buffer, opt.match_thresh, opt.mot20_check, opt.frame_rate)
error_logging.info("--> Model is loaded successfully!" )


## ini camera & setting
cap = ipcamCapture()


# event = threading.Event()
#####################################################
##  Camera Start + (async) AI Inference + return top Queue
#####################################################

@app.post('/camera_start', tags=["IPcam Method"])
async def camera_start(previos_time: Item_prevtime):
    try:
        cam_start, reset_ai = False, []
        today = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_video_path = opt.output_video_path +today+'.avi'
        time.sleep(0.5)
        ## Camera Start
        # if opt.video_test == True:
        #     input_video_path = "C:/CTOS_DAI/Bytetrack_yolov7/video/fastapi/TBN4223132_1_20220610-153705_Trim2.mp4"
        #     cap = cv2.VideoCapture(input_video_path)
        #     AI_Inference = threading.Thread(target=Yolov7, daemon=True, args=(opt, cap, deteted, output_video_path,)).start()

        # else:
        global stop_threads, AI_Inference
        # error_logging.info(str(cap.isstop))
        if cap.isstop == True:
            try:
                cap.start(opt.cam_ip)
                # cam_start_ = func_timeout(10, cap.start, args=(opt.cam_ip,), kwargs=None)
            except FunctionTimedOut:
                if cap.isstop == True:
                    e = 'Camera open is Timeout : %s (s)' % str(20)
                    error_logging.error('!!! IPcamera is disconnected & '+str(e))
                    return ApiResponse(result=False, error='!!! IPcamera is disconnected'+str(e)) 
                else:
                    e = 'Camera open is Timeout : %s (s)' % str(20)
                    error_logging.error(e)
                    return ApiResponse(result=False, error=str(e))
                
            cam_start = True
            replyData='--> Camera turn-on Sucessfully!'
            error_logging.info(replyData)

            ## (if measure_base =1) Check previos time
            
            previos_time = previos_time.dict()
            _previos_time = json.loads(json.dumps(previos_time))
            CMD_previos_time = PrevTime_chk(_previos_time, False)
            error_logging.info('Script CMD: '+ str(_previos_time))

            ## (async) AI Inference
            
            try:
                stop_threads = False
                if opt.track_id == "True":
                    AI_Inference = threading.Thread(target=Yolov7_ByteTrack, daemon=True, args=(opt, deteted, tracker, output_video_path,)).start()
                    # AI_Inference.setDaemon(True) # thread will close after main closed.
                    # AI_Inference.start()
                else:
                    AI_Inference = threading.Thread(target=Yolov7, daemon=True, args=(opt, deteted, output_video_path, CMD_previos_time,)).start()
                    # AI_Inference.setDaemon(True) # thread will close after main closed.
                    # AI_Inference.start()
            except Exception as e:
                error_logging.error(e)

            return ApiResponse(result=cam_start, replyData=replyData)
        else:
            try:

                previos_time = previos_time.dict()
                _previos_time = json.loads(json.dumps(previos_time))
                CMD_previos_time = PrevTime_chk(_previos_time, False)
                replyData='--> Camera reset! AI FPS : ' + str(CMD_previos_time)
                error_logging.info(replyData)
            
            except Exception as e:
                error_logging.error(e)

            try:
                # error_logging.info('camera stop !')
                cap.stop()
                # cap = ipcamCapture()
                time.sleep(0.1)
                

            except:
                error_logging.error('camera stop err !!')
            

            if cap.isstop == True:
                try:
                    # vid_writer.release()
                    # cv2.destroyAllWindows()
                    
                    cap.start(opt.cam_ip)
                    # cam_start_ = func_timeout(10, cap.start, args=(opt.cam_ip,), kwargs=None)
                    time.sleep(1)
                    error_logging.info('camera start !')
                    
                except FunctionTimedOut:
                    if cap.isstop == True:
                        e = 'Camera open is Timeout : %s (s)' % str(20)
                        error_logging.error('!!! IPcamera is disconnected & '+str(e))
                        return ApiResponse(result=False, error='!!! IPcamera is disconnected'+str(e)) 
                    else:
                        e = 'Camera open is Timeout : %s (s)' % str(20)
                        error_logging.error(e)
                        return ApiResponse(result=False, error=str(e))
                    

                ## (async) AI Inference
                try:
                    stop_threads = False
                    if opt.track_id == "True":
                        AI_Inference = threading.Thread(target=Yolov7_ByteTrack, daemon=True, args=(opt, deteted, tracker, output_video_path,)).start()
                        # AI_Inference.setDaemon(True) # thread will close after main closed.
                        # AI_Inference.start()
                    else:
                        AI_Inference = threading.Thread(target=Yolov7, daemon=True, args=(opt, deteted, output_video_path, CMD_previos_time,)).start()
                        # AI_Inference.setDaemon(True) # thread will close after main closed.
                        # AI_Inference.start()
                except Exception as e:
                    error_logging.error(e)
            else:
                print('Camera is not stop !!')
            error_logging.warning(replyData)
            return ApiResponse(result=True, replyData=replyData)

    except Exception as e:
        error_logging.error(e)
        return ApiResponse(result=False, error=e)

#####################################################
##  Global queue to DAI player
#####################################################

@app.get('/AI_result', tags=["Get AI_result Method"])
async def AI_result():
    try:
        if cap.isstop == False:
            if len(G_queue) != 0:
                replyData='--> return AI results are Sucessfully!'
                # error_logging.info(replyData)
                return ApiResponse(result=G_queue, replyData=replyData)
        else:
            replyData='--> PLZ turn on camera first! '
            error_logging.warning(replyData)
            return ApiResponse(result=False, replyData=replyData)

    except Exception as e:
        error_logging.error(e)
        return ApiResponse(result=False, error=e)

#####################################################
##  pic_capture
#####################################################
def img_cap(img_path):
    _,im0 = cap.read_()
    imgpath = img_path.dict()
    imgpath_ = json.loads(json.dumps(imgpath))
    imgpath_ = imgpath_['img_path'].replace('\\', '/')
    head_ = os.path.split(imgpath_)[0]
    if not os.path.isdir(head_):
        os.mkdir(head_)
    
    cv2.imwrite(imgpath_,im0)
    
@app.post('/pic_capture', tags=["POST pic_capture Method"])
async def pic_capture(img_path: Item_path):
    try:
        if cap.isstop == False:
            
            img_capture = threading.Thread(target=img_cap, daemon=True, args=(img_path,)).start()
            replyData='--> Image capture is Sucessfully! Image path : ' + str(img_path)
            error_logging.info(replyData)
            return ApiResponse(result=True, replyData=replyData)
        else:
            replyData='--> PLZ turn on camera first! '
            error_logging.warning(replyData)
            return ApiResponse(result=False, replyData=replyData)

    except Exception as e:
        error_logging.error(e)
        return ApiResponse(result=False, error=e)



#####################################################
##  Get queue + Check cmd + Check rule + return 0/1
#####################################################

@app.post('/check_obj', tags=["Obj_check Method"])
def check_obj(CMD: Item):
    try:
        l_ = []
        t0 = time.time()
        ## Check - script cmd
        CMD = CMD.dict()
        _CMD = json.loads(json.dumps(CMD))
        CMD_LabelName, CMD_StartupPoint, CMD_EndPoint, CMD_ObjectCount, \
        CMD_CheckFlag, CMD_TimeOut, CMD_ContinuousTime, CMD_IouMode, CMD_MeasureBased\
        = CMD_Check(_CMD, False)
        error_logging.info(' * Script CMD: '+ str(_CMD))
        
        ## Get queue
        Result_chk, Result_chk_, replyData, _Result_chk_ = False, False, False, False

        ## Check - rule
        AIList = L_List()
        
        if CMD_MeasureBased == 0: ## based on cmd time
            
            while float(time.time()-t0) < CMD_ContinuousTime:
                print(time.time())
                print(float(time.time()-t0))
                time.sleep(0.1)
                try:
                    # AI_obj_res, label_list, res_iou_list_
                    Result_chk, detect_label, detect_iou, detect_num = func_timeout(CMD_TimeOut, AI_Object_chk, args=(
                                                                            CMD_LabelName, 
                                                                            CMD_StartupPoint, 
                                                                            CMD_EndPoint, 
                                                                            CMD_ObjectCount, 
                                                                            CMD_CheckFlag,
                                                                            CMD_ContinuousTime,
                                                                            CMD_IouMode
                                                                        ), kwargs=None)
                    
                except FunctionTimedOut:
                    if cap.isstop == True:
                        e = 'AI Object check Function is Timeout : %s (s)' % str(CMD_TimeOut)
                        error_logging.error('!!! IPcamera is disconnected & '+str(e))
                        return ApiResponse(result=False, error='!!! IPcamera is disconnected') 
                    else:
                        e = 'AI Object check Function is Timeout : %s (s)' % str(CMD_TimeOut)
                        error_logging.error(e)
                        return ApiResponse(result=False, error=e)
                
                ## Result_chk == True 
                    # 1. True: end if correspond with contiuous time
                    # 2. False: end if other wise.
                if Result_chk == True:
                    AIList.push_(Result_chk)
                    Result_chk_ = AIList.chk_CMD_ContinuousTime(l_, CMD_ContinuousTime)
                    
                    try:
                        print(record_)
                        if len(record_) < 2 :
                            if Result_chk_ == None:
                                replyData='=== First correspond with cmd result ! ==='
                                t0 = int(time.time())
                                continue

                            else:
                                _Result_chk_ = Result_chk_.data
                                if t_false != 0:
                                    t0 = int(time.time())
                                    continue

                                if _Result_chk_ == True:
                                    # replyData='--> Continuous time check result : ' + str(True)
                                    replyData = 'Detect label : ' + str(list(detect_label)) + ' ,and labels are in IOU : ' + str(detect_iou) + ',and detect object num is : ' + str(detect_num)
                                    error_logging.info(' * AI results -- ' + replyData)
                                    continue
                                else:
                                    replyData='--> Continuous time check result : ' + str(False)
                                    error_logging.info(' * AI results -- ' + replyData)
                                    continue   
                        else:
                            if Result_chk_ == None or record_[0]!=record_[1]:
                                
                                replyData='=== First correspond with cmd result ! ==='
                                t0 = int(time.time())
                                continue
                            
                            else:
                                _Result_chk_ = Result_chk_.data
                                if t_false != 0:
                                    t0 = int(time.time())
                                    continue

                                if _Result_chk_ == True:
                                    # replyData='--> Continuous time check result : ' + str(True)
                                    replyData = 'Detect label : ' + str(list(detect_label)) + ' ,and labels are in IOU : ' + str(detect_iou) + ',and detect object num is : ' + str(detect_num)
                                    error_logging.info(' * AI results -- ' + replyData)
                                    continue
                                else:
                                    replyData='--> Continuous time check result : ' + str(False)
                                    error_logging.info(' * AI results -- ' + replyData)
                                    continue   

                    except Exception as e:
                        error_logging.error(e)
                        return ApiResponse(result=False, error=e)   
            
        if CMD_MeasureBased == 1: ## based on previos time
            try:
                _Result_chk_, detect_label, detect_iou, detect_num = func_timeout(CMD_TimeOut, AI_Object_chk, args=(
                                                                                CMD_LabelName, 
                                                                                CMD_StartupPoint, 
                                                                                CMD_EndPoint, 
                                                                                CMD_ObjectCount, 
                                                                                CMD_CheckFlag,
                                                                                CMD_ContinuousTime,
                                                                                CMD_IouMode
                                                                            ), kwargs=None)
                    
                replyData='--> Based on previos time : ' + str(fps) + '(s) is True.'
                error_logging.info(replyData)

            except FunctionTimedOut:
                if cap.isstop == True:
                    e = 'AI Object check Function is Timeout : %s (s)' % str(CMD_TimeOut)
                    error_logging.error('!!! IPcamera is disconnected & '+str(e))
                    return ApiResponse(result=False, error='!!! IPcamera is disconnected') 
                else:
                    e = 'AI Object check Function is Timeout : %s (s)' % str(CMD_TimeOut)
                    error_logging.error(e)
                    return ApiResponse(result=False, error=e)

        return ApiResponse(result=_Result_chk_, replyData=replyData)

    except Exception as e:
        error_logging.error(e)
        return ApiResponse(result=False, error=e)


#####################################################
##  Camera stop + check ipcam_isstop + release loop
#####################################################

@app.get('/camera_end', tags=["IPcam Method"])
async def camera_end():
    try:
        cam_end = False
        cap.stop()
        time.sleep(0.5)
        
        if cap.isstop == True:
            cam_end = True
            replyData='--> Camera is closed Sucessfully!'
            error_logging.info(replyData)  

        return ApiResponse(result=cam_end, replyData=replyData) 
        
        
    except Exception as e:
        error_logging.error(e)
        return ApiResponse(result=False, error=e)

from starlette.responses import FileResponse
@app.post('/predict_video', tags=["POST Method"])
async def predict_video(video: UploadFile = File(...)):
    try:
        # Save video
        try:
            input_video_path = './video/fastapi/'
            output_video_path = opt.output_video_path + video.filename
            
            if not os.path.exists(opt.output_video_path):
                os.makedirs(opt.output_video_path)

            if not os.path.exists(input_video_path):
                os.makedirs(input_video_path)
                
            with Path(input_video_path+video.filename).open("wb") as buffer:
                shutil.copyfileobj(video.file, buffer)
        finally:
            video.file.close()

        # process video
        Yolov7_video(opt, deteted, input_video_path+video.filename, output_video_path)

        return FileResponse(output_video_path)
        # return StreamingResponse(iterfile(output_video_path), media_type="video/mp4")
    
    
    except Timeout as e :
        return ApiResponse(result=False, error='Timeout error')	
    except ApplicationError as e:
        return ApiResponse(result=False, error=e)
    except Exception as e:
        return ApiResponse(result=False, error=e)
        # return ApiResponse(result=False, error='unexpected server error')


if __name__ == '__main__':
    uvicorn.run(app, host=opt.ip,port=opt.port, debug=True)
