from typing import List
from frame import FrameInterface, Frame, GpuFrame
import cv2 as cv
import numpy as np
import copy

matcher = cv.BFMatcher()
# matcher = cv.cuda.DescriptorMatcher_createBFMatcher()


def get_residual(src, dst, inner, M):
    A = M[:, 0:2]
    t = M[:, 2]

    losts = []
    assert(len(src) == len(dst))
    assert(len(inner) == len(dst))

    for i in range(len(src)):
        if inner[i] != 1:
            continue
        src_pt = src[i]
        dst_pt = dst[i]
        pt = np.dot(A, src_pt) + t - dst_pt
        error = pt[0] * pt[0] + pt[1] * pt[1]
        losts.append(error)
    
    return sum(losts) / len(losts)
        

class FrameSequenceInterface(FrameInterface):
    def __init__(self):
        super.__init__(self)
    def add_frame(self, frame, match_ratio=0.6, max_residual=3):
        pass
class GpuFrameSequence(FrameSequenceInterface):
    def __init__(self):
        FrameInterface.__init__(self)
        self.m_frame_list: List[Frame] = []

    def __estimate_rigid_transform(self, frame:Frame, max_residual):
        list = self.m_frame_list
        frame1 = list[-1]
        frame2 = frame

        kp1, desc1 = frame1.get_kp(), frame1.get_kp_description()
        kp2, desc2 = frame2.get_kp(), frame2.get_kp_description()

        if desc1 is None or desc2 is None:
            return None
        
        matcher = cv.BFMatcher()
        # matcher = cv.cuda.DescriptorMatcher_createBFMatcher()
        matches = matcher.knnMatch(desc2, desc1, k=2)

        # good = []
        # for m, n in matches:
        #     if m.distance < match_ratio * n.distance:
        #         good.append(m)
        good = [m for m, n in matches]

        src_pts = np.float32([kp2[m.queryIdx].pt for m in good])
        dst_pts = np.float32([kp1[m.trainIdx].pt for m in good])

        if len(src_pts) == 0 or len(dst_pts) == 0:
            return None

        src_pts = src_pts.reshape((-1, 2))
        dst_pts = dst_pts.reshape((-1, 2))

        rigid_m, _ = cv.estimateAffinePartial2D(src_pts, dst_pts)

        residual = get_residual(src_pts, dst_pts, rigid_m)
        if residual >= max_residual:
            return None
        return rigid_m

    def add_frame(self, frame: Frame, match_ratio=0.6, max_residual=3):
        list = self.m_frame_list

        if len(list) == 0:
            list.append(frame)
            return True

        # else:
        rigid_m = self.__estimate_rigid_transform(frame, max_residual)

        R1, _, T = cv.decomposeEssentialMat(rigid_m)

        rotation = list[-1].get_R()
        transfer = list[-1].get_T()
        rotation = np.dot(R1, rotation)
        transfer = T + transfer
        frame.set_R(rotation)
        frame.set_T(transfer)
        list.append(frame)
        return True

class GpuFrameSequence(FrameSequenceInterface):
    def __init__(self):
        FrameInterface.__init__(self)
        self.m_frame_list: List[GpuFrame] = []
    
    def __estimate_rigid_transform(self, frame1:GpuFrame, frame2:GpuFrame, max_residual):
        kp1, desc1 = frame1.get_kp(), frame1.get_gpu_kp_description()
        kp2, desc2 = frame2.get_kp(), frame2.get_gpu_kp_description()

        if desc1 is None or desc2 is None:
            return None
        
        matcher = cv.cuda.DescriptorMatcher_createBFMatcher()
        matches = matcher.knnMatch(desc2, desc1, k=2)

        # good = []
        # for m, n in matches:
        #     if m.distance < match_ratio * n.distance:
        #         good.append(m)
        good = [m for m, n in matches]

        src_pts = np.float32([kp2[m.queryIdx].pt for m in good])
        dst_pts = np.float32([kp1[m.trainIdx].pt for m in good])

        if len(src_pts) == 0 or len(dst_pts) == 0:
            return None

        src_pts = src_pts.reshape((-1, 2))
        dst_pts = dst_pts.reshape((-1, 2))

        rigid_m, inner = cv.estimateAffinePartial2D(src_pts, dst_pts)

        residual = get_residual(src_pts, dst_pts, inner, rigid_m)
        if residual >= max_residual:
            return None
        return rigid_m

    def add_frame(self, frame: GpuFrame, match_ratio=0.6, max_residual=3):
        list = self.m_frame_list

        if (len(list) == 0):
            list.append(frame)
            return True
        
        last_frame = list[-1]
        rigid_m = self.__estimate_rigid_transform(last_frame, frame, max_residual)
        if rigid_m is None:
            return False
        proj_m = np.vstack([rigid_m, [0, 0, 1]])

        # 通常来讲上一帧只需要和当前帧做匹配，所以这里就可以gpu缓存了
        # 当前帧的不进行清除，gpu缓存留到下一帧
        last_frame.clear_gpu_cache()
        M = last_frame.get_M()
        M = np.dot(M, proj_m)
        frame.set_M(M)
        list.append(frame)
        return True