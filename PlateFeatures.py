 #!/usr/bin/env python
#coding=gbk
import cv2
import numpy
import Function

class PlateFeatures(object):
    def __init__(self):
        pass

    #获取直方图信息
    def getHistogramFeatures(self, img):
        gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_thershold = cv2.threshold(gray_image, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        return Function.Func.features(img_thershold[1],0)

