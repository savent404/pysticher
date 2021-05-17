from frame import Frame

class FrameSequence:
    def __init__(self):
        self.m_frame_list = []

    def add_frame(self, frame:Frame):
        list = self.m_frame_list

        if len(list) == 0:
            list.append(frame)
            return True
        else:
            return False
