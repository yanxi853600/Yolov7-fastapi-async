import mmap
import cv2
import numpy as np

class SharedMemory_Image:

    def __init__(self,_Name):
        self.shape = (1080, 1920, 3)
        self.Name = _Name
        self.mm = mmap.mmap(-1, np.prod(self.shape), _Name)
        
    def ReleaseMemoryMapped(self):
        try:
            self.mm.close()
        except:
            return False
        return True

    def WriteMemoryMapped(self,_Image):
        try:
            if self.mm is None:
                self.mm = mmap.mmap(-1, np.prod(self.shape), self.Name)
            img = cv2.resize(_Image, list(reversed(self.shape[:-1])), interpolation = cv2.INTER_AREA)
            buffer = img.tobytes()
            self.mm.seek(0)
            self.mm.write(buffer)
            self.mm.flush()
            return True
        except:
            return False

    def ReadMemoryMapped(self):
        try:
           
            mm = mmap.mmap(-1, np.prod(self.shape), self.Name)
            mm.seek(0)
            buf = mm.read(np.prod(self.shape))
            mm.close()

            #img = np.frombuffer(buf, dtype=np.uint8).reshape(mm.shape)

            return True, buf
        except:
            return False




