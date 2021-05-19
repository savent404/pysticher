import numpy as np


class CameraParameter:
    def __init__(self, inner=np.eye(3), outer=np.eye(4), calibration=None):
        self.inner = inner
        self.outer = outer
        self.calibration = calibration
