import numpy as np
import cv2 as cv
from cameraParameter import CameraParameter

class Frame:
    def __init__(self, image = np.array((1, 1, 3), dtype=np.uint8), camPara=CameraParameter()):
        self.m_R = camPara.outer[:3, :3]
        self.m_T = camPara.outer[:3, 3]
        self.m_img = image
        self.m_kp = None
        self.m_desc = None

    def get_R(self):
        '''
        获取旋转矩阵
        '''
        return self.m_R
    
    def get_T(self):
        '''
        获取平移矩阵
        '''
        return self.m_T
    
    def get_image(self):
        '''
        获取图像本身
        '''
        return self.m_img
    
    def __get_keypoints_and_description(self):
        detector = cv.xfeatures.SIFT_create()
        gray = cv.cvtColor(self.m_img, cv.COLOR_BGR2GRAY)

        _, mask = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
        kp, desc = detector.detectAndCompute(gray, mask)
        return kp, desc

 
    def get_kp(self):
        if self.m_desc is None:
            self.__get_keypoints_and_description()
        return self.m_kp
    
    def get_kp_description(self):
        if self.m_desc is None:
            self.__get_keypoints_and_description()
        return self.m_desc


