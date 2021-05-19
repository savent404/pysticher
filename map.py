import cv2
import numpy as np
import math
import copy
from algo import get_padding_transform, merge_image, gpu_merge_image


class MapBlock:
    '''
    存储栅格化地图的最小单位
    保存图像信息以及位置信息(抽象)
    '''

    def __init__(self, size, pos):
        assert(type(pos[0] == 'int'))
        assert(type(pos[1] == 'int'))
        s = (size[0], size[1], 1)
        image = np.zeros(s, np.uint8)
        self.image = image
        self.x = pos[0]
        self.y = pos[1]


class Map:
    '''
    可扩展栅格化地图
    坐标系遵从opencv参考坐标
    -----------------> x
    | 0,0 | 1,0 |
    |------------
    | 0,1 | 1,1 |
    |------------
    | y

    '''

    def __init__(self, size=(1000, 1000, 3), offset=(0, 0)):
        self.__blocks = []
        self.__block_offset = [0, 0]
        self.__block_size = size
        self.__pixel_offset = np.reshape(offset, (2,))
        self.__resized = True
        self.__cached_map = None
        self.__blocks.append(
            MapBlock(size, (self.__block_offset[0], self.__block_offset[1])))

    def __pixel_toVirtualSpace(self, x, y):
        '''
        将真实空间坐标转为程序空间坐标
        计算减去地图像素偏移量的相对像素位置
        '''
        x = x - self.__pixel_offset[0]
        y = y - self.__pixel_offset[1]
        return np.array((x, y))

    def __index_toVirtualSpace(self, x, y):
        '''
        将真实空间坐标转为程序空间坐标
        将真实世界的index(有负值）转为程序世界的index（无负值）
        NOTE: Map会根据传入的负平面坐标对整个坐标系的block_offset进行更新，并更新所有block块的xy坐标
        '''
        x = x - self.__block_offset[0]
        y = y - self.__block_offset[1]
        if x < 0:
            self.__block_offset[0] = self.__block_offset[0] + x
            for b in self.__blocks:
                b.x = b.x - x
            x = 0
            self.__resized = True

        if y < 0:
            self.__block_offset[1] = self.__block_offset[1] + y
            for b in self.__blocks:
                b.y = b.y - y
            y = 0
            self.__resized = True
        return np.array((x, y))

    def __getMapBlock_VirtualSpace(self, x, y):
        '''
        x, y应为非负值
        根据block索引得到对应的block map
        '''
        assert(x >= 0 and y >= 0)
        for b in self.__blocks:
            if b.x == x and b.y == y:
                return b
        b = MapBlock(self.__block_size, (x, y))
        self.__blocks.append(b)
        self.__resized = True
        return b

    def getMapOffsetTransfer(self):
        '''
        获取当前地图的offset
        '''
        bsize = self.__block_size
        return np.float32([
            [1, 0, -self.__block_offset[0] * bsize[0]],
            [0, 1, -self.__block_offset[1] * bsize[1]],
            [0, 0, 1]
        ])

    def genMap(self, limit=None, debug=False):
        if limit is None:
            return self.__genMap_full(debug)
        else:
            return self.__genMap_partial(limit, debug)

    def __genMap_partial(self, limit, debug=False):
        bsize = self.__block_size
        start = limit[0]
        end = limit[1]

        x1, y1 = self.__pixel_toVirtualSpace(start[0], start[1])
        x2, y2 = self.__pixel_toVirtualSpace(end[0], end[1])

        start = [math.floor(x1 / bsize[0]), math.floor(y1 / bsize[1])]
        end = [math.floor(x2 / bsize[0]), math.floor(y2 / bsize[1])]

        sx, sy = x1 - start[0] * bsize[0], y1 - start[1] * bsize[1]
        ex, ey = x2 - start[0] * bsize[0], y2 - start[1] * bsize[1]

        image = self.__getImage_RealSpace(start, end)
        image = image[sy:ey, sx:ex]

        if debug:
            h, w = image.shape[:2]
            originX = -x1
            originY = -y1
            color = (0, 255, 255)
            image = cv2.line(image, (0, originY), (w, originY), color, thickness=10)
            image = cv2.line(image, (originX, 0), (originX, h), color, thickness=10)

            raws = end[1] - start[1]
            cols = end[0] - start[0]
            color = (0, 128, 128)

            for r in range(raws + 1):
                for c in range(cols + 1):
                    x, y = c * bsize[0] - sx, r * bsize[1] - sy
                    image = cv2.line(image, (0, y), (w, y), color, thickness=2)
                    image = cv2.line(image, (x, 0), (x, h), color, thickness=2)
        return image

    def __genMap_full(self, debug=False):
        '''
        生成全景地图

        地图会根据:
        - limit指定的位置
        - 地图的总大小
        来生成地图
        '''
        bsize = self.__block_size
        maxX = 0
        minX = 0
        maxY = 0
        minY = 0
        for b in self.__blocks:
            x, y = b.x, b.y
            if x > maxX:
                maxX = x
            elif x < minX:
                minX = x
            if y > maxY:
                maxY = y
            elif y < minY:
                minY = y

        # 重新获取完整图像
        if self.__resized is True:
            img = self.__getImage_VirtualSpace((minX, minY), (maxX, maxY))
            self.__cached_map = img
            self.__resized = False
            for b in self.__blocks:
                b.modified = False

        # 更新block局部值
        for b in self.__blocks:
            if b.modified is False:
                continue
            xStart = bsize[0] * b.x
            xEnd = xStart + bsize[0]
            yStart = bsize[1] * b.y
            yEnd = yStart + bsize[1]
            image = self.__cached_map
            assert(yStart >= 0 and yEnd <= image.shape[0])
            assert(xStart >= 0 and xEnd <= image.shape[1])
            image[yStart:yEnd, xStart:xEnd] = b.image
            self.__cached_map = image
            b.modified = False
        # 提供拷贝版本
        img = copy.copy(self.__cached_map)

        # 画坐标参考线
        if debug:
            h, w = img.shape[:2]
            offsetX = self.__block_offset[0] * self.__block_size[0]
            offsetY = self.__block_offset[1] * self.__block_size[1]
            color = (0, 255, 255)
            img = cv2.line(img, (0, -offsetY), (w, -offsetY),
                        color, thickness=10)
            img = cv2.line(img, (-offsetX, 0), (-offsetX, h),
                        color, thickness=10)
        # 画block区域线
        if debug:
            raws = maxY - minY
            cols = maxX - minX
            size = self.__block_size
            color = (0, 128, 128)
            h, w = img.shape[:2]

            for r in range(raws):
                for c in range(cols):
                    img = cv2.line(
                        img, (0, r * size[1]), (w, r*size[1]), color, thickness=2)
                    img = cv2.line(
                        img, (c * size[0], 0), (c * size[1], h), color, thickness=2)
        return img

    def __getImage_VirtualSpace(self, start, end):
        '''
        根据程序空间坐标合并图像
        '''
        assert(start[0] >= 0 and start[1] >= 0)
        assert(end[0] >= 0 and end[1] >= 0)
        bsize = self.__block_size
        start = np.reshape(start, (2,))
        end = np.reshape(end, (2,))
        d = end - start
        w = (d[0] + 1) * bsize[0]
        h = (d[1] + 1) * bsize[1]
        c = bsize[2]

        mat = np.zeros((h, w, c), np.uint8)
        for i in range(d[0] + 1):
            I = start[0] + i
            for j in range(d[1] + 1):
                J = start[1] + j
                block = self.__getMapBlock_VirtualSpace(I, J)
                block.modified = True
                xStart = i * bsize[0]
                xEnd = xStart + bsize[0]
                yStart = j * bsize[1]
                yEnd = yStart + bsize[1]
                mat[yStart:yEnd, xStart:xEnd] = block.image
        return mat

    def __getGpuImage_VirtualSpace(self, start, end):
        '''
        根据程序空间坐标合并图像
        '''
        assert(start[0] >= 0 and start[1] >= 0)
        assert(end[0] >= 0 and end[1] >= 0)
        bsize = self.__block_size
        start = np.reshape(start, (2,))
        end = np.reshape(end, (2,))
        d = end - start
        w = (d[0] + 1) * bsize[0]
        h = (d[1] + 1) * bsize[1]
        c = bsize[2]

        mat = cv2.cuda_GpuMat(h, w, cv2.CV_8UC3)
        for i in range(d[0] + 1):
            I = start[0] + i
            for j in range(d[1] + 1):
                J = start[1] + j
                block = self.__getMapBlock_VirtualSpace(I, J)
                block.modified = True
                xStart = i * bsize[0]
                xEnd = xStart + bsize[0]
                yStart = j * bsize[1]
                yEnd = yStart + bsize[1]

                img = cv2.cuda_GpuMat()
                img.upload(block.image)
                roi = cv2.cuda_GpuMat(mat, (yStart, yEnd), (xStart, xEnd))
                img.copyTo(roi)
        return mat

    def __getImage_RealSpace(self, start, end):
        '''
        根据真实空间坐标合并图像
        '''
        start = self.__index_toVirtualSpace(start[0], start[1])
        end = self.__index_toVirtualSpace(end[0], end[1])
        return self.__getImage_VirtualSpace(start, end)
    
    def __getGpuImage_RealSpace(self, start, end):
        '''
        根据真实空间坐标合并图像
        '''
        start = self.__index_toVirtualSpace(start[0], start[1])
        end = self.__index_toVirtualSpace(end[0], end[1])
        return self.__getGpuImage_VirtualSpace(start, end)


    def __putImage(self, img, start, end):
        '''
        按照方块区域更新该区域图像

        start=(0, 0)
        end=(1, 0)
        '''
        bsize = self.__block_size
        start = self.__index_toVirtualSpace(start[0], start[1])
        end = self.__index_toVirtualSpace(end[0], end[1])
        d = end - start
        for i in range(d[0] + 1):
            I = start[0] + i
            for j in range(d[1] + 1):
                J = start[1] + j
                block = self.__getMapBlock_VirtualSpace(I, J)
                xStart = i * bsize[0]
                xEnd = xStart + bsize[0]
                yStart = j * bsize[1]
                yEnd = yStart + bsize[1]
                block.image = img[yStart:yEnd, xStart:xEnd]
    
    def __putGpuImage(self, img, start, end):
        '''
        按照方块区域更新该区域图像

        start=(0, 0)
        end=(1, 0)
        '''
        bsize = self.__block_size
        start = self.__index_toVirtualSpace(start[0], start[1])
        end = self.__index_toVirtualSpace(end[0], end[1])
        d = end - start
        for i in range(d[0] + 1):
            I = start[0] + i
            for j in range(d[1] + 1):
                J = start[1] + j
                block = self.__getMapBlock_VirtualSpace(I, J)
                xStart = i * bsize[0]
                xEnd = xStart + bsize[0]
                yStart = j * bsize[1]
                yEnd = yStart + bsize[1]

                roi = cv2.cuda_GpuMat(img, (yStart, yEnd), (xStart, xEnd))
                block.image = roi.download()


    # def merge_enforce(self, other):
    #     other_img = other.genMap()
    #     other_transfer = other.getMapOffsetTransfer()
    #     x, y = other_transfer[0, 2], other_transfer[1, 2]
    #     self.update(other_img, (-x, -y), order=2)
    #     pass

    # def merge(self, other, residual=10):
    #     self_img = self.genMap()
    #     other_img = other.genMap()
    #     matrix, _ = calculateMatrixByFeatureDectection(self_img, other_img, request_residual=None)
    #     if matrix is not None:
    #         offset = self.getMapOffsetTransfer()
    #         other_offset = other.getMapOffsetTransfer()
    #         other_offset_inv = np.linalg.inv(other_offset)
    #         move_transform = np.dot(offset, other_offset_inv)
    #         image, transfer = warp_img(other_img, np.dot(move_transform, matrix))

    #         move_transform = np.dot(transfer, move_transform)
    #         x, y = move_transform[0, 2], move_transform[1, 2]
    #         self.update(image, (x, y))
    #     else:
    #         return False

    def update(self, img, pos, order = 1):
        '''
        根据utm绝对位置，自动将新图片叠加到对应区域
        - pos: 图像左上角对应utm位置
        - order 1 override self
                2 override other
        '''
        bsize = self.__block_size
        pos = np.int32(pos)
        x, y = self.__pixel_toVirtualSpace(pos[0], pos[1])
        img_h, img_w = img.shape[:2]

        start = [math.floor(x / bsize[0]), math.floor(y / bsize[1])]
        end = [math.floor((x + img_w) / bsize[0]),
               math.floor((y+img_h)/bsize[1])]

        sx, sy = x - start[0] * bsize[0], y - start[1] * bsize[1]
        ex, ey = sx + img_w, sy + img_h

        combine = self.__getImage_RealSpace(start, end)
        bigger_img = np.zeros(combine.shape, dtype=np.uint8)
        bigger_img[sy:ey, sx:ex] = img
        if order == 1:
            combine = merge_image(bigger_img, combine)
        elif order == 2:
            combine = merge_image(combine, bigger_img)
        else:
            raise NotImplementedError
        self.__putImage(combine, start, end)
    
    def update_gpu(self, img, pos, order = 1):
        '''
        根据utm绝对位置，自动将新图片叠加到对应区域
        - pos: 图像左上角对应utm位置
        - order 1 override self
                2 override other
        '''
        bsize = self.__block_size
        pos = np.int32(pos)
        x, y = self.__pixel_toVirtualSpace(pos[0], pos[1])
        img_w, img_h = img.size()

        start = [math.floor(x / bsize[0]), math.floor(y / bsize[1])]
        end = [math.floor((x + img_w) / bsize[0]),
               math.floor((y+img_h)/bsize[1])]

        sx, sy = x - start[0] * bsize[0], y - start[1] * bsize[1]
        ex, ey = sx + img_w, sy + img_h

        combine = self.__getGpuImage_RealSpace(start, end)
        c_w, c_h = combine.size()
        bigger_img = cv2.cuda_GpuMat(c_h, c_w, cv2.CV_8UC3)
        roi = cv2.cuda_GpuMat(bigger_img, (sy, ey), (sx, ex))
        img.copyTo(roi)
        if order == 1:
            combine = gpu_merge_image(bigger_img, combine)
        elif order == 2:
            combine = gpu_merge_image(combine, bigger_img)
        else:
            raise NotImplementedError
        self.__putGpuImage(combine, start, end)
