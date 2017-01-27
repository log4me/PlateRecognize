#!/usr/bin/env python
#coding=gbk
import cv2
import numpy as np
import PlateDetect
import CharRecognize
from Util import ID
class PlateRecognize(object):
    def __init__(self):
        self.plate_detect = PlateDetect.PlateDetect()
        self.char_recognize = CharRecognize.CharRecognize()

    def plate_recognize(self, img_name):
        src_img = cv2.imread(img_name)

        plate_results = self.plate_detect.plate_detect(src_img)
        if len(plate_results) <= 0:
            return []
        res = []
        for i in range(len(plate_results)):
            plate_info = {}
            plate = plate_results[i]
            plate_name = 'images/temp/plate/plate{0}{1}.jpg'.format(i,ID.get_ID())
            plate_info['name'] = plate_name
            cv2.imwrite('static/' + plate_name, plate)
            plate_num = self.char_recognize.chars_recognize(plate)
            plate_info['num'] = plate_num
            if plate_num == '':
                print("Error:{} 字符识别错误".format(plate_name))
                res.append(plate_info)
            else:
                res.append(plate_info)
        return res




if __name__ == '__main__':
    plate_recognize = PlateRecognize()
    res = plate_recognize.plate_recognize('C28888.jpg')
    if res == False:
        print("No plate")
    else:
        for i in res:
            print('{0}:{1}'.format(i['name'],i['num']))
