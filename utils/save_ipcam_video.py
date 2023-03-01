from IP_Camera import ipcamCapture
from func_timeout import func_timeout, FunctionTimedOut
import cv2
from datetime import datetime

def save():    
    print('ipcam is start to save video!')
    while True:
        _, img = cap.read_()
        # plot_img_h = int(img.shape[0]/2.5 )
        # plot_img_w= int(img.shape[1]/2.5)
        # plot_img = cv2.resize(img, (plot_img_w, plot_img_h), interpolation=cv2.INTER_AREA)

        # cv2.namedWindow('DAI_Inference', 0)
        # cv2.resizeWindow('DAI_Inference', (plot_img_w, plot_img_h))
        # cv2.moveWindow("DAI_Inference", 0, 0)
        # cv2.imshow("DAI_Inference", img)
        # ch = cv2.waitKey(1)  # 1 millisecond

        # if ch == ord("q") : 
        #     break

        vid_writer.write(img)

today = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
url = "rtsp://169.254.25.13//video0.sdp"
output = "C:/home/ims/yolov7/Bytetra_yolov7/bkp/2022_12_05/video/M3_Screw/ipcam/6FPS/90CM/degree_0/diff/"+today+'.avi'

cap = ipcamCapture()
cap.start(url)  

fps, w, h = cap.getvideotype()
vid_writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'XVID'), 30, (w, h))

try:
    ipcam_video = func_timeout(10, save, args=(), kwargs=None)
except FunctionTimedOut:
    cap.stop()
    vid_writer.release()

