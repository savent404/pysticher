from cameraParameter import CameraParameter
from frame import Frame, GpuFrame
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
        fileName = dir + '/pos.txt'
        imageDirectory = dir + '/images/'

        DOFs = np.genfromtxt(fileName, delimiter=",",
                             usecols=range(1, 7), dtype=float)
        imgList = np.genfromtxt(fileName, delimiter=",",
                                usecols=[0], dtype=str)
        imgList = ["{}/{}".format(imageDirectory, iter) for iter in imgList]

        default_inner_matrix = np.float32([[1399.008608021883, 0, 912.8436694988062], [
                                          0, 1398.067750138546, 560.9715646208826], [0, 0, 1]])
        default_camera_calibration = np.float32([-0.1928987635480564, 1.185777385330833, -
                                                 0.008740735312992649, -0.00720632424414056, -4.413429081461387])
        default_camera_calibration = None # 畸变参数不准确，忽略畸变矫正
        camParam = CameraParameter(
            inner=default_inner_matrix, calibration=default_camera_calibration)

        self.m_method = "simulate"
        self.m_DOFs = DOFs
        self.m_img_list = imgList
        self.m_index = 0
        self.m_cam_param = camParam

    def get_frame(self, isGpu=False):
        if self.is_eof():
            return None

        img = None
        while img is None:
            index = self.m_index
            img_path = self.m_img_list[index]
            dof = self.m_DOFs[index]
            img = cv.imread(img_path)
            img = self.__calibration(img)
            self.m_index = self.m_index + 1
            cam_param = copy.copy(self.m_cam_param)
        
        if isGpu:
            return GpuFrame(img, cam_param)
        else:
            return Frame(img, cam_param)

    def is_eof(self):
        '''
        判断是否需要结束
        '''
        return self.m_index >= len(self.m_img_list)

    def __calibration(self, image):
        '''
        修正图像畸变
        '''
        k = self.m_cam_param.inner
        d = self.m_cam_param.calibration

        h, w = image.shape[:2]
        mapx, mapy = cv.initUndistortRectifyMap(
            k, d, None, None, (w, h), cv.CV_32FC1)
        return cv.remap(image, mapx, mapy, cv.INTER_LINEAR)
