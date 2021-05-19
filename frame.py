import numpy as np
import cv2 as cv
import copy
from cameraParameter import CameraParameter
from algo import get_padding_transform


class FrameInterface:
    def __init__(self):
        self.m_img = None
        self.m_kp = None
        self.m_desc = None
        self.m_R = None
        self.m_T = None
        self.m_M = None

    def get_M(self):
        '''
        获取完整映射矩阵
        '''
        # return np.dot(self.m_R, self.m_T)
        return self.m_M

    def set_M(self, M):
        self.m_M = M
        self.m_R = copy.copy(M)
        self.m_R[0, 2] = 0
        self.m_R[1, 2] = 0

        self.m_T = np.eye(3)
        self.m_T[0, 2] = M[0, 2]
        self.m_T[1, 2] = M[1, 2]

    def get_R(self):
        '''
        获取旋转矩阵
        '''
        return self.m_R

    def set_R(self, R):
        '''
        更新旋转矩阵
        '''
        self.m_R = R

    def get_T(self):
        '''
        获取平移矩阵
        '''
        return self.m_T

    def set_T(self, T):
        '''
        设置平移矩阵
        '''
        self.m_T = T

    def get_image(self):
        '''
        获取图像本身
        '''
        return self.m_img

    def get_rotated_image(self):
        '''
        根据R矩阵得到的图像
        '''
        raise NotImplemented

    def set_image(self, img):
        self.m_img = img

    def get_kp(self):
        return self.m_kp

    def get_kp_description(self):
        return self.m_desc

    def merge(self, frame):
        pass

    def clear_cache(self):
        '''
        清除缓存（即为了加速自动添加到class上的内容）
        '''
        pass

    def decomposeM(self, M):
        # r, _, t = cv.decomposeEssentialMat(M)
        r = copy.copy(M)
        t = np.eye(3, dtype=np.float32)
        r[0, 2] = 0
        r[1, 2] = 0
        t[0, 2] = M[0, 2]
        t[1, 2] = M[1, 2]
        return r, t


class Frame(FrameInterface):
    def __init__(self, image=np.array((1, 1, 3), dtype=np.uint8), camPara=CameraParameter()):
        FrameInterface.__init__(self)
        self.m_R = camPara.outer[:3, :3]
        self.m_T = camPara.outer[:3, 3]
        self.m_M = camPara.outer[0:3, 0:3]
        self.m_img = image
        self.m_kp = None
        self.m_desc = None

    def set_image(self, img):
        '''
        设置图像
        '''
        self.m_desc = None
        self.m_kp = None
        # self.m_img = img
        FrameInterface.set_image(self, img)

    def __get_detector(self):
        return cv.xfeatures2d.SURF_create(5)

    def clear_cache(self):
        self.m_kp = None
        self.m_desc = None
        super().clear_cache()

    def _get_kp_and_desc(self):
        detector = self.__get_detector()
        img = self.m_img
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        _, mask = cv.threshold(gray, 1, 255, cv.THRESH_BINARY)
        kp, desc = detector.detectAndCompute(gray, mask)

        # write cache
        self.m_kp = kp
        self.m_desc = desc
        return kp, desc

    def get_kp(self):
        if self.m_desc is None:
            self._get_kp_and_desc()
        return FrameInterface.get_kp(self)

    def get_kp_description(self):
        if self.m_desc is None:
            self._get_kp_and_desc()
        return FrameInterface.get_kp_description(self)


class GpuFrame(Frame):
    def __init__(self, img, camPara=CameraParameter()):
        FrameInterface.__init__(self)
        super().set_M(camPara.outer[0:3, 0:3])
        self.m_kp = None
        self.m_desc = None
        self.m_img = img
        self.m_gpu_desc = None
        self.m_gpu_img = None

    def _get_kp_and_desc(self):
        detector = self.__get_detector()
        img = self.get_gpu_img()
        gray = cv.cuda.cvtColor(img, cv.COLOR_BGR2GRAY)
        _, mask = cv.cuda.threshold(gray, 1, 255, cv.THRESH_BINARY)
        kp, desc = detector.detectWithDescriptors(gray, mask)
        # write cache
        self.m_kp = detector.downloadKeypoints(kp)
        self.m_gpu_desc = desc
        return kp, desc

    def __get_detector(self):
        return cv.cuda.SURF_CUDA_create(5)

    def get_gpu_img(self):
        if self.m_gpu_img is None:
            a = cv.cuda_GpuMat()
            a.upload(self.m_img)
            self.m_gpu_img = a
        return self.m_gpu_img

    def clear_cache(self):
        self.m_gpu_desc = None
        self.m_gpu_img = None
        return super().clear_cache()

    def get_kp_description(self):
        if self.m_gpu_desc is None or self.m_desc is None:
            self._get_kp_and_desc()
        if self.m_desc is None:
            self.m_desc = self.m_gpu_desc.download()
        return self.m_desc

    def get_gpu_kp_description(self):
        if self.m_gpu_desc is None:
            self._get_kp_and_desc()
        return self.m_gpu_desc
