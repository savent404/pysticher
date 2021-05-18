from typing import List
from frame import FrameInterface, Frame
import cv2 as cv
import numpy as np
import copy

matcher = cv.BFMatcher()
# matcher = cv.cuda.DescriptorMatcher_createBFMatcher()


def get_residual(src, dst, M):
    return 1


class FrameSequence(FrameInterface):
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

        rigid_m = np.eye(3)

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
