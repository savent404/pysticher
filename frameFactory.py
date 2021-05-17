from cameraParameter import CameraParameter
from frame import Frame
import numpy as np
import cv2 as cv
import copy

class FrameFactory:
    def get_frame():
        return Frame()

    def is_eof():
        return True

class SimulateFrameFactory(FrameFactory):
    def __init__(self, dir='datasets'):
        '''
        创建实例
        '''
        fileName =  dir + '/pos.txt'
        imageDirectory = dir + '/images/'

        DOFs = np.genfromtxt(fileName, delimiter=",", usecols=range(1, 7), dtype=float)
        imgList = np.genfromtxt(fileName, delimiter=",", usecols=[0], dtype=str)
        imgList = [ "{}/{}".format(imageDirectory, iter) for iter in imgList]

        default_inner_matrix = [[1399, 0, 912.84], [0, 1398, 560.97, 0, 0, 1]]
        camParam = CameraParameter(inner=default_inner_matrix)

        self.m_method = "simulate"
        self.m_DOFs = DOFs;
        self.m_img_list = imgList
        self.m_index = 0
        self.m_cam_param = camParam

    def get_frame(self): 
        if self.is_eof():
            return None

        img = None
        while img is None:
            index = self.m_index
            img_path = self.m_img_list[index]
            dof = self.m_DOFs[index]
            img = cv.imread(img_path)
            self.m_index = self.m_index + 1
            cam_param = copy.copy(self.m_cam_param)
        return Frame(img, cam_param)
        
    def is_eof(self):
        '''
        判断是否需要结束
        '''
        return self.m_index >= len(self.m_img_list)