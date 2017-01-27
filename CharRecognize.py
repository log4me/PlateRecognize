#!/usr/bin/env python
#coding=gbk
import cv2
import numpy
import CharSegment
import CharIdentify
import CharMatch
class CharRecognize(object):
    def __init__(self):
        self.char_segment = CharSegment.CharSegment()
        self.char_identify = CharIdentify.CharIdentify()

    #车牌字符识别，识别整个车牌的字符
    def chars_recognize(self, plate):
        plate_chars = ''
        result = self.char_segment.char_segment(plate)
        if result == False :
            return  ''
        i = 0
        for char_img in result:
            plate_chars += self.char_identify.char_identify(char_img,(0 == i),(1 == i))
            i += 1
        return plate_chars
