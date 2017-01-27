#!/usr/bin/env python
#coding=gbk
import cv2
import numpy as np
import math
class Func(object):
    color = dict()
    color['UNKNOWN'] = 0
    color['BLUE'] = 1
    color['YELLOW'] = 2
    direction = dict()
    direction['UNKNOWN'] = 0
    direction['VERTICAL'] = 1
    direction['HORIZONTAL'] = 2
    @classmethod
    def colorMatch(cls, src, color_r, minsv):
        '''
        根据图像和颜色模板，获取对应的二值图
        :param color_r: 颜色模板，蓝色或者黄色
        :param minsv: 如果是True:则最小值取决于H，按比例衰减，否则使用固定值
        :param src: 输入RGB图
        :return:
        '''
        max_sv = 255.0
        minref_sv = 64.0
        minabs_sv = 95.0

        #蓝色的H范围
        min_blue = 100
        max_blue = 140
        #黄色的H范围
        min_yellow = 15
        max_yellow = 40
        #转到HSV空间进行处理，使用H分量进行颜色匹配
        src_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(src_hsv)
        r = cv2.equalizeHist(v)
        src_hsv = cv2.merge((h, s, v))
        #匹配模板基色
        min_H = 0
        max_H = 0
        if color_r == cls.color['BLUE']:
            min_H = min_blue
            max_H = max_blue
        elif color_r == cls.color['YELLOW']:
            min_H = min_yellow
            max_H = max_yellow
        diff_H = (max_H - min_H)/2.0
        avg_H = int(min_H + diff_H)
        rows, cols, channels = src_hsv.shape
        for i in range(rows):
            for j in range(cols):
                H = src_hsv[i][j][0] & 0xFF
                S = src_hsv[i][j][1] & 0xFF
                V = src_hsv[i][j][2] & 0xFF
                colorMatched = False
                if H > min_H and H < max_H:
                    H_diff = 0
                    if H > avg_H :
                        H_diff = int(H - avg_H)
                    else :
                        H_diff = int(avg_H - H)

                    H_diff_p = H_diff * 1.0 / diff_H

                    min_sv = 0.0
                    if minsv :
                        min_sv = minref_sv - minref_sv * 1.0 / 2 * (1- H_diff_p)
                    else :
                        min_sv = minabs_sv
                    if ((S > min_sv and S <= max_sv) and (V > min_sv and V < max_sv)):
                        colorMatched = True
                if colorMatched == True:
                    src_hsv[i][j][0] = 0
                    src_hsv[i][j][1] = 0
                    src_hsv[i][j][2] = 255
                else:
                    src_hsv[i][j][0] = 0
                    src_hsv[i][j][1] = 0
                    src_hsv[i][j][2] = 0
        h, s, v = cv2.split(src_hsv)
        return v

    @classmethod
    def plateColorJudge(cls, src, color_r, minsv):
        '''
        :param src: 车牌矩阵
        :param color_r: 颜色模板
        :param minsv: 同上
        :return:
        '''
        thresh = 0.49
        gray = cls.colorMatch(src, color_r, minsv)
        row, col = gray.shape
        percent = cv2.countNonZero(gray) * 1.0 / gray.size
        if percent > thresh:
            return True
        else:
            return False

    @classmethod
    def getHistogram(cls, src, direction):
        '''
        获取水平或存执方向直方图
        :param direction:
        :return:
        '''
        sz = 0
        if direction == cls.direction['HORIZONTAL']:
            sz = src.shape[0]
        elif direction == cls.direction['VERTICAL']:
            sz = src.shape[1]
        nonZeorMat = []
        img = cv2.extractChannel(src,0)
        for i in range(sz):
            data = None
            if direction == cls.direction['HORIZONTAL']:
                data = img[i]
            else :
                data = img[:,i]
            count = cv2.countNonZero(data)
            nonZeorMat.append(count)

        max_ = max(nonZeorMat)
        if max_ > 0 :
            for i in range(len(nonZeorMat)):
                nonZeorMat[i] /= 1.0 * max_
        return nonZeorMat

    @classmethod
    def features(cls, src, sizeData):
        '''
        样本特征为水平，垂直直方图和低分辨路图像所组成的向量
        :param src:
        :param sizeData:
        :return:
        '''
        vhist = cls.getHistogram(src, cls.direction['VERTICAL'])
        hhist = cls.getHistogram(src, cls.direction['HORIZONTAL'])
        lowData = np.ndarray((0,0))
        if sizeData > 0 :
            lowData = cv2.resize(src, (sizeData,sizeData))
        numCols = len(vhist) + len(hhist) + lowData.size
        out = np.zeros((1,numCols),np.float32)
        j = 0
        for i in range(len(vhist)):
            out[0][j] = vhist[i]
            j += 1
        for i in range(len(hhist)):
            out[0][j] = hhist[i]
            j += 1
        row, col = lowData.shape
        for i in range(col):
            for k in range(row):
                val = float(lowData[k][i] & 0xFF)
                out[0][j] = val
                j += 1

        return out

    @classmethod
    def get_plate_type(cls, plate_src, minsv):
        if cls.plateColorJudge(plate_src, cls.color['BLUE'], minsv) == True:
            return cls.color['BLUE']
        elif cls.plateColorJudge(plate_src, cls.color['YELLOW'], minsv) == True:
            return cls.color['YELLOW']
        else:
            return cls.color['UNKNOWN']


    @classmethod
    def get_pixel_feture(cls, char_src):
        return char_src.reshape(-1)
        out = np.zeros((1,char_src.size),np.float32)
        char_src = char_src.reshape(1,-1)
        col = char_src.shape[1]
        for i in range(col):
            out[0][i] = char_src[0][i]
        return out