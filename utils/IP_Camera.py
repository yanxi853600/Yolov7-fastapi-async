from pickle import TRUE
import cv2
import time
import threading
from func_timeout import func_timeout, FunctionTimedOut, func_set_timeout


class ipcamCapture:

    def __init__(self):
        self.capture = None
        self.status = False
        self.isstop = True	
        self.Frame = []
        self.start_thread = None

    @func_set_timeout(20)
    def start(self,URL):
        try:
            print('ipcam started!')
            self.capture = cv2.VideoCapture(URL)
            #self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
            self.isstop = False
            self.start_thread = threading.Thread(target=self.queryframe, daemon=True, args=())
            self.start_thread.start()
            time.sleep(0.3)
            return self.isstop
        except Exception as e:
            print(e)

    @func_set_timeout(10)
    def stop(self):
        try:
            time.sleep(0.1)
            self.isstop = True
            self.start_thread.join()
            # self.capture.release()
            # print('capture.release()')
            # time.sleep(0.1)
            # self.Frame = []
            # print('Frame = []')
            # time.sleep(0.1)
            # self.isstop = True
            print('isstop = True')
        except Exception as e:
            print(e)

    # @func_set_timeout(1)
    def read_(self):
        try:
            if self.Frame.size != 0:
                return True, self.Frame.copy()
        except:
            return False, []

    def getvideotype(self):
        try:
            fps = self.capture.get(cv2.CAP_PROP_FPS)
            w = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return fps, w, h
        except Exception as e:
            print(e)

    # @func_set_timeout(1)
    def cv2_read(self):
        try:
            self.status, self.Frame =self.capture.read()
        except Exception as e:
            print(e)
        
    def queryframe(self):
        try:
            while (not self.isstop):
                self.cv2_read()
                # self.status, self.Frame = self.capture.read()
        except FunctionTimedOut:
            self.stop()
