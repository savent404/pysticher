import threading
import debug
import copy
from typing import List
from frameFactory import FrameFactory
from frameSequence import GpuFrameSequence
from viewer import Viewer
from loguru import logger
import cv2 as cv
import numpy as np


class sticher_runner(threading.Thread):
    def __init__(self, factory: FrameFactory, frameseq_list: List[GpuFrameSequence], viewer: Viewer):
        threading.Thread.__init__(self)
        self.m_factory = factory
        self.m_seq_list = frameseq_list
        self.m_viewer = viewer
        pass

    def run(self):
        factory = self.m_factory
        seq_list = self.m_seq_list
        seq = seq_list[-1]

        while factory.is_eof() is not True:
            timer = debug.DebugTimer()
            frame = factory.get_frame(isGpu=True)
            if seq.add_frame(frame, max_residual=100) is False:
                # 残差过大导致无法合入
                # create a new seq
                logger.info("new frame sequence")
                seq = GpuFrameSequence()
                self.m_seq_list.append(seq)
                seq.add_frame(frame)

            # write to cache map
            self.m_viewer.add_gpu_frame(frame)
            self.m_viewer.do_view()
            # clear pre frames cached memory
            for s in seq_list[:-1]:
                s.clear_cache()
            for f in seq.m_frame_list[:-1]:
                f.clear_cache()
            logger.debug("duration: {}".format(timer.duration()))
            # Debug
            debug_img = copy.copy(frame.get_image())
            cv.drawKeypoints(frame.get_image(), frame.get_kp(), debug_img)
            debug.display('preFrame', debug_img)
