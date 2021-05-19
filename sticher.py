from typing import List
from frameFactory import FrameFactory, SimulateFrameFactory
from frameSequence import GpuFrameSequence
from viewer import Viewer
from loguru import logger
from sticherRunner import sticher_runner


class Sticher:
    def __init__(self, factories: List[FrameFactory]):
        self.m_factories = factories
        self.m_frameseq: List[GpuFrameSequence] = []
        self.m_viewer = Viewer()
        for _ in range(len(factories)):
            frame_list = []
            frame_list.append(GpuFrameSequence())
            self.m_frameseq.append(frame_list)

        self.m_runners = []
        for i in range(len(factories)):
            runner = sticher_runner(
                self.m_factories[i], self.m_frameseq[i], self.m_viewer)
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
        raise NotImplemented

    def download_map(self, file_path: str):
        '''
        下载地图到文件
        '''
        raise NotImplemented
        # cv.imwrite(file_path, self.get_map())


if __name__ == '__main__':
    ff = SimulateFrameFactory()

    sticher = Sticher([ff])
    sticher.run()
    sticher.download_map('./result.jpg')
