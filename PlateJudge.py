#!/usr/bin/env python
#coding=gbk
import cv2
import numpy
import os
import PlateFeatures
class PlateJudge(object):
    def __init__(self):
        self.svm = cv2.SVM()
        self.svm.load("res/model/svm.xml")
        self.features = PlateFeatures.PlateFeatures()

    def plate_judge(self, img):
        features = self.features.getHistogramFeatures(img)
        features = features.reshape(-1)
        features.dtype = 'float32'
        ret = self.svm.predict(features)
        return int(ret)

    def plate_judge_mul(self, srcs):
        plate_res = []
        for img in srcs:
            if self.plate_judge(img) == 1:
                plate_res.append(img)
            else:
                width = img.shape[1]
                height = img.shape[0]
                copy = img.copy()
                copy = copy[int(height * 0.1):int(height * 0.9),int(width * 0.1):int(width * 0.9)]
                resize_plate = cv2.resize(copy, (width, height), interpolation=cv2.INTER_CUBIC)
                if self.plate_judge(resize_plate) == 1 :
                    plate_res.append(resize_plate)
        return plate_res
if __name__ == '__main__':
    plate_judge = PlateJudge()
    img = cv2.imread("temp/16.jpg")
    ret = plate_judge.plate_judge(img)
    print ret
    for i in range(45):
        img = cv2.imread("temp/{}.jpg".format(i))
        print("pic {0}:{1}".format(i, plate_judge.plate_judge(img)))