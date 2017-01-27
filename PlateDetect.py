#!/usr/bin/env python
#coding=utf-8
import cv2
import numpy as np
import platelocate
import PlateJudge
class PlateDetect(object):
    def __init__(self):
        self.plate_locate = platelocate.PlateLocate()
        self.plate_locate.set_mode()
        self.plate_judge = PlateJudge.PlateJudge()

    def plate_detect(self, src_img):
        plate_imgs = self.plate_locate.plate_locate(src_img)
        plate_imgs = self.plate_judge.plate_judge_mul(plate_imgs)
        return plate_imgs