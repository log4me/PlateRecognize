#!/usr/bin/env python
#coding=gbk
#find plate in img
import cv2
from cv2 import *
import cv2.cv as cv
import numpy as np
from Util import ID
class PlateLocate(object):
    def __init__(self):
        #车牌定位所需变量
        self.gaussian_blur_size = 5
        self.sobel_scale = 1
        self.sobel_delta = 0
        self.sobel_depth = cv2.CV_16S
        self.sobel_x_weight = 1
        self.sobel_y_weight = 0
        self.morph_size_width = 17
        self.morph_size_height = 3
        #车牌判断所需变量
        self.verify_min = 3
        self.verify_max = 20
        self.verify_error = 0.9
        self.ratio = 3.142857
        self.angle = 30
        #车牌调整所需变量
        self.width = 136
        self.height = 36
        self.type = cv2.CV_8UC3
    def set_mode(self):
        self.gaussian_blur_size = 5
        self.morph_size_width = 9
        self.morph_size_height = 3
        self.verify_error = 0.9
        self.ratio = 4
        self.verify_max = 30
        self.verify_min = 1

    def plate_locate(self, src):
        '''
            定位车牌
            :param src : source image
            :return : list with all possible plate
        '''

        result_rects = []
        if src is not None:
            h = src.shape[0]
            w = src.shape[1]
            if (h > 50):
                src = src[0:h-50]
        #高斯模糊
        gaus_blur = cv2.GaussianBlur(src, (self.gaussian_blur_size,self.gaussian_blur_size), 0, 0, cv2.BORDER_DEFAULT)
        cv2.imwrite("static/images/temp/platelocate/gaussian_blur{}.jpg".format(ID.get_ID()), gaus_blur)

        #灰度化
        gray = cv2.cvtColor(gaus_blur, cv2.COLOR_RGB2GRAY)
        cv2.imwrite("static/images/temp/platelocate/gray{}.jpg".format(ID.get_ID()), gray)

        #Sobel运算，得到水平方向导数
        grad_x = cv2.Sobel(gray, self.sobel_depth, 1, 0, ksize=3, scale=self.sobel_scale, delta=self.sobel_delta,borderType=cv2.BORDER_DEFAULT)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        grad_y = cv2.Sobel(gray, self.sobel_depth, 0, 1, ksize = 3, scale=self.sobel_scale, delta=self.sobel_delta, borderType=cv2.BORDER_DEFAULT)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        grad = cv2.addWeighted(abs_grad_x,self.sobel_x_weight, abs_grad_y, self.sobel_y_weight, 0)
        cv2.imwrite("static/images/temp/platelocate/sobel{}.jpg".format(ID.get_ID()), grad)

        #二值化
        _, img_threshold = cv2.threshold(grad, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        cv2.imwrite("static/images/temp/platelocate/threshold{}.jpg".format(ID.get_ID()), img_threshold)

        #闭操作
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (self.morph_size_width, self.morph_size_height))
        img_mophology = cv2.morphologyEx(img_threshold, cv2.MORPH_CLOSE, element)
        cv2.imwrite("static/images/temp/platelocate/morphology{}.jpg".format(ID.get_ID()), img_mophology)

        #提取轮廓
        src_back = src.copy()
        contours,_ = cv2.findContours(img_mophology,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(src, contours, -1, (0, 255, 0))
        imwrite("static/images/temp/platelocate/contours{}.jpg".format(ID.get_ID()), src)

        #判断外接矩形大小，进行筛选
        boxs = []
        res_plate = []
        for contour in contours:
            min_outer_rect = cv2.minAreaRect(contour)# min_outer_rect = ((center_x,center_y),(width,height),angle)
            #area = cv2.contourArea(contour)
            box = cv2.cv.BoxPoints(min_outer_rect)
            box = np.int0(box)
            if self.verify_by_size(min_outer_rect):
                result_rects.append(min_outer_rect)
                boxs.append(box)
                ys = [box[0, 1], box[1, 1], box[2, 1], box[3, 1]]
                xs = [box[0, 0], box[1, 0], box[2, 0], box[3, 0]]
                ys_sorted_index = np.argsort(ys)
                xs_sorted_index = np.argsort(xs)

                x1 = box[xs_sorted_index[0], 0]
                x2 = box[xs_sorted_index[3], 0]

                y1 = box[ys_sorted_index[0], 1]
                y2 = box[ys_sorted_index[3], 1]
                plate_rect = src_back[y1:y2, x1:x2]
                #统一大小
                if len(plate_rect)  > 0 and len(plate_rect[0] > 0):
                    resize_plate = cv2.resize(plate_rect,(self.width, self.height),interpolation=cv2.INTER_CUBIC)

                res_plate.append(resize_plate)
        for i in range(len(res_plate)):
            cv2.imwrite("static/images/temp/platelocate/rectangle{}.jpg".format(i), res_plate[i])

        cv2.drawContours(src, boxs, -1, (0,0,255))
        imwrite("static/images/temp/platelocate/contours_rectangle{}.jpg".format(ID.get_ID()), src)
        return res_plate
        #角度处理 TODO



    def set_gaussian_blur_size(self, blur_radius):
        '''
        set the radius of gaussian blur
        :param blur_radius: blur radius
        :return: None
        '''
        pass

    def set_morph_size_width(self, morph_size):
        '''
        set size of morph
        :param morph_size: size
        :return:None
        '''
        pass

    def set_verify_error(self, error):
        '''
        set verify error
        :param error:
        :return: None
        '''
        pass

    def set_side_ratio(self, ratio):
        '''
        ratio is length/width
        :param ratio:
        :return: None
        '''
        pass

    def set_verify_max(self, verify_max):
        '''
        set max size of plate
        :param verify_max:
        :return: None
        '''
        pass

    def set_verify_min(self, verify_min):
        '''
        set min size of plate
        :param verify_min:
        :return: None
        '''
        pass

    def set_judge_angle(self, angle):
        '''
        set angle can receive,plate angle between -angle and angle will be accept
        :param angle:
        :return: None
        '''
        pass

    def verify_by_size(self, rect):
        '''
        fliter the rectangle by area and ratio
        :param rect:
        :return:True if accept
        '''
        #我国车牌大小为440mm * 140mm, ratio = 3.142857
        # 我国车牌大小为440mm * 140mm, ratio = 3.142857
        min = 44 * 14 * self.verify_min
        max = 44 * 14 * self.verify_max
        # 允许误差
        rmin = self.ratio - self.ratio * self.verify_error
        rmax = self.ratio + self.ratio * self.verify_error
        height = rect[1][1]
        width = rect[1][0]
        if height == 0 or width == 0 :
            return False
        area = height * width
        rect_ratio = height / width
        if rect_ratio < 1 :
            rect_ratio = width * 1.0  / height
        return area >= min and area <= max and rect_ratio >= rmin and rect_ratio <= rmax



if __name__ == '__main__':
    src = cv2.imread("test.jpg")

    plate_locate = PlateLocate()
    plate_locate.set_mode()
    plate_locate.plate_locate(src)