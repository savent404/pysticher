from typing import List
from frameFactory import FrameFactory
from frameSequence import FrameSequence
import cv2 as cv
import numpy as np
import threading
import debug
import copy

class sticher_runner(threading.Thread):
    def __init__(self, factory: FrameFactory, frameseq_list: List[FrameSequence]):
        threading.Thread.__init__(self)
        self.m_factory = factory
        self.m_seq_list = frameseq_list
        pass

    def run(self):
        factory = self.m_factory
        seq_list = self.m_seq_list
        seq = seq_list[-1]

        while factory.is_eof() is not True:
            timer = debug.DebugTimer()
            frame = factory.get_frame()
            if seq.add_frame(frame) is False:
                # create a new seq
                seq = FrameSequence()
                self.m_seq_list.append(seq)
                seq.add_frame(frame)
            print("duration: {}".format(timer.duration()))
            # Debug
            debug_img = copy.copy(frame.get_image())
            cv.drawKeypoints(frame.get_image(), frame.get_kp(), debug_img)
            debug.display('preFrame', debug_img)

class Sticher:
    def __init__(self, factories: List[FrameFactory]):
        self.m_factories = factories
        self.m_frameseq: List[FrameSequence] = []
        for _ in range(len(factories)):
            frame_list = []
            frame_list.append(FrameSequence())
            self.m_frameseq.append(frame_list)

        self.m_runners = []
        for i in range(len(factories)):
            runner = sticher_runner(self.m_factories[i], self.m_frameseq[i])
            self.m_runners.append(runner)

    def run(self):
        for r in self.m_runners:
            r.start()

        for r in self.m_runners:
            r.join()

    def get_map(self, is_final=False):
        '''
        获取地图
        '''
        return np.uint8((1, 1, 3))

    def download_map(self, file_path: str):
        '''
        下载地图到文件
        '''
        cv.imwrite(file_path, self.get_map())
