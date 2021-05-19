import cv2 as cv
from algo import gpu_get_padding_transform
from map import Map
from frame import Frame, GpuFrame
import threading
import numpy as np
import debug

class Viewer:
    def __init__(self):
        self.previwe_map = Map()
        self.lock = threading.Lock()
    
    def add_frame(self, frame:Frame):
        raise NotImplementedError

    def add_gpu_frame(self, frame:GpuFrame):
        '''

        由于图像投影（旋转+平移）后需要进行平移确保所有图像均在投影屏幕内，需要叠加一个move平移矩阵
        则由frame叠加到map的完整公式为
        Y = move_inv * T * move * R * X
        '''
        self.lock.acquire()
        R, T = frame.get_R(), frame.get_T()
        img = frame.get_gpu_img()

        move, size= gpu_get_padding_transform(img, transform=R)
        warp_m = np.dot(move, R)
        warped = cv.cuda.warpAffine(img, warp_m[0:2, :], size)
        warped = warped.download()

        move_inv = np.linalg.inv(move)
        warp_T = np.dot(move_inv, T)
        x, y = warp_T[0, 2], warp_T[1, 2]
        pos = (x, y)
        self.previwe_map.update(warped, pos)
        self.lock.release()

    def do_view(self):
        debug.display('map', self.previwe_map.genMap(debug=True))
