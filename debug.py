import cv2 as cv
import time
def display(title, image, timeout=1, trueSize=False):
    '''
    OpenCV machinery for showing an image until the user presses a key.
    :param title: Window title in string form
    :param image: ndArray containing image to show
    :return:
    '''
    cv.namedWindow(title, cv.WINDOW_NORMAL)
    if trueSize:
        cv.resizeWindow(title, image.shape[1], image.shape[0])
    cv.imshow(title, image)
    cv.waitKey(timeout)

class DebugTimer:
    def __init__(self):
        self.m_start = time.time()
        self.m_stop = None
    def start(self):
        self.m_start = time.time()
    def stop(self):
        self.m_stop = time.time()
    def duration(self):
        if self.m_stop is None:
            self.stop()
        return self.m_stop - self.m_start