import cv2 as cv
import numpy as np

def __get_padding_transform(corners, transform):
    A, M = np.hsplit(transform, [2])
    warpedCorner = np.zeros((4, 2))
    for i in range(0, 4):
        # T = A*x + M
        point = np.reshape(corners[i], (2, 1))
        res = np.dot(A, point) + M
        warpedCorner[i] = res.reshape((1, 2))

    # all = np.concatenate((warpedCorner), axis=0)
    all = warpedCorner

    [xMin, yMin] = np.int32(all.min(axis=0).ravel() - 0.5)
    [xMax, yMax] = np.int32(all.max(axis=0).ravel() + 0.5)

    size = (xMax - xMin, yMax - yMin)
    transform = np.float32(([1, 0, -xMin], [0, 1, -yMin], [0, 0, 1]))

    return transform, size

def get_padding_transform(image1, image2=None, transform=np.eye(3)):
    '''
    计算两张图像拼接后图像的大小以及两张图像所需的位移矩阵(3x3, 仅有两个自由度)
    Return:
    - [0]->transform: padding变化
    - [1]->size: 合并图像大小
    '''
    transform = transform[:2, :].reshape((2, 3))
    h1, w1 = image1.shape[:2]

    corner1 = np.float32(([0, 0], [0, h1], [w1, h1], [w1, 0]))
    if image2 is not None:
        h2, w2 = image2.shape[:2]
        corner2 = np.float32(([0, 0], [0, h2], [w2, h2], [w2, 0]))
        corners = np.vstask([corner1, corner2])
    else:
        corners = corner1
    return __get_padding_transform(corners, transform)

def gpu_get_padding_transform(image1, image2=None, transform=np.eye(3)):
    '''
    输入图像为GpuMat
    '''
    transform = transform[:2, :].reshape((2, 3))
    w1, h1 = image1.size()

    corner1 = np.float32(([0, 0], [0, h1], [w1, h1], [w1, 0]))
    if image2 is not None:
        w2, h2 = image2.size()
        corner2 = np.float32(([0, 0], [0, h2], [w2, h2], [w2, 0]))
        corners = np.vstask([corner1, corner2])
    else:
        corners = corner1
    return __get_padding_transform(corners, transform)

def merge_image(base, target):
    '''
    在base基础上叠加target
    '''
    baseGray = cv.cvtColor(base, cv.COLOR_BGR2GRAY)
    targetGray = cv.cvtColor(target, cv.COLOR_BGR2GRAY)
    _, baseMaskInv = cv.threshold(baseGray, 1, 255, cv.THRESH_BINARY_INV)
    _, targetMask = cv.threshold(targetGray, 1, 255, cv.THRESH_BINARY)
    baseMaskInvBigger = cv.dilate(baseMaskInv, np.ones((6, 6), dtype=np.uint8), 1)
    overlapMask = cv.bitwise_xor(baseMaskInv, baseMaskInvBigger)
    overlapMask = cv.bitwise_and(overlapMask, targetMask)
    overlayMaskInv = cv.bitwise_not(overlapMask)

    t1 = cv.bitwise_and(target, target, mask=baseMaskInv)
    t2 = cv.bitwise_and(target, target, mask=overlapMask)

    dst = cv.bitwise_or(base, t1)
    dst = cv.bitwise_and(dst, dst, mask=overlayMaskInv)
    dst = cv.bitwise_or(dst, t2)
    return dst

def gpu_merge_image(base, target):
    '''
    在base基础上叠加target
    '''
    baseGray = cv.cuda.cvtColor(base, cv.COLOR_BGR2GRAY)
    targetGray = cv.cuda.cvtColor(target, cv.COLOR_BGR2GRAY)
    _, baseMaskInv = cv.cuda.threshold(baseGray, 1, 255, cv.THRESH_BINARY_INV)
    _, targetMask = cv.cuda.threshold(targetGray, 1, 255, cv.THRESH_BINARY)
    _baseMaskInvBigger = cv.dilate(baseMaskInv.download(), np.ones((6, 6), dtype=np.uint8), 1)
    baseMaskInvBigger = cv.cuda_GpuMat()
    baseMaskInvBigger.upload(_baseMaskInvBigger)
    overlapMask = cv.cuda.bitwise_xor(baseMaskInv, baseMaskInvBigger)
    overlapMask = cv.cuda.bitwise_and(overlapMask, targetMask)
    overlayMaskInv = cv.cuda.bitwise_not(overlapMask)

    target = cv.cuda.split(target)
    base = cv.cuda.split(base)

    t1 = [cv.cuda.bitwise_and(m, m, mask=baseMaskInv) for m in target]
    t2 = [cv.cuda.bitwise_and(m, m, mask=overlapMask) for m in target]

    dst = []
    for i in range(3):
        m = cv.cuda.bitwise_or(base[i], t1[i])
        m = cv.cuda.bitwise_and(m, m, mask=overlayMaskInv)
        m = cv.cuda.bitwise_or(m, t2[i])
        dst.append(m)
    res = cv.cuda_GpuMat()
    res.upload(cv.cuda.merge(dst))
    return res

