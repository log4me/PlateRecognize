#!/usr/bin/env python
#coding=gbk
import cv2
import numpy as np
import Function
import random
class CharSegment(object):
    def __init__(self):
        self.char_size = 20
        self.horizontal = 1
        self.vertical = 0
        self.rivet_size = 7
        self.mat_width = 136
        self.color_threshold = 150
        self.blue_percent = 0.3
        self.white_percent = 0.1

    def char_segment(self, plate_img):
        if len(plate_img) == 0 or len(plate_img[0]) == 0:
            return
        #灰度化
        plate_grey = cv2.cvtColor(plate_img, cv2.COLOR_RGB2GRAY)

        width = plate_img.shape[1]
        height = plate_img.shape[0]
        #去除车牌边框的影响
        temp_plate = plate_img[int(height * 0.1):int(height *0.9),int(width * 0.1):int(width * 0.9)]
        #去除铆钉,按颜色不同选择不同的灰度化选项
        if Function.Func.get_plate_type(temp_plate, True) == Function.Func.color['BLUE']:
            _,img_threshold = cv2.threshold(plate_grey, 10, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        elif Function.Func.get_plate_type(temp_plate, True) == Function.Func.color['YELLOW']:
            _,img_threshold = cv2.threshold(plate_grey, 10, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
        else:
            _, img_threshold = cv2.threshold(plate_grey, 10, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
        cv2.imwrite('static/images/temp/charsegment/车牌二值化.jpg', img_threshold)
        #去除车牌上方以及下方干扰
        img_threshold = self.clear_rivet(img_threshold)
        cv2.imwrite('static/images/temp/charsegment/去除铆钉.jpg', img_threshold)
        #找字符轮廓
        plate_contours = img_threshold.copy()
        contours,_ = cv2.findContours(plate_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        res_rect = []
        for contour in contours :
            rect = cv2.boundingRect(contour)
            x, y, w, h = rect
            char = img_threshold[y-2:y+h+2,x:x+w]
            if self.verify_size(char):
                res_rect.append(rect)
        if len(res_rect) <= 0:
            return False
        #排序
        sorted_rect = sorted(res_rect, key = lambda x:x[0])
        #得到城市标识
        spec_indx = self.get_speci_rect(sorted_rect)
        if spec_indx >= len(sorted_rect):
            return False
        x, y, w, h = sorted_rect[spec_indx]
        spec_char = img_threshold[y:y+h,x:x+w]
        cv2.imwrite("static/images/temp/charsegment/代表字母.jpg", spec_char)
        #获取中文字符的Rect
        chinese_rect = self.get_chinese_rect(sorted_rect[spec_indx])
        x, y, w, h = chinese_rect
        chinese_char = img_threshold[y:y+h,x:x+w]
        #chinese_char = plate_img[y:y+h,x:x+w]
        cv2.imwrite("static/images/temp/charsegment/汉字.jpg", chinese_char)
        chars_rect = self.rebuild_rect(sorted_rect,spec_indx,chinese_rect)
        i = 0
        chars_img = []
        ran = random.randint(1,100)
        for char_rect in chars_rect :
            x, y, w, h = char_rect
            char = img_threshold[y:y+h,x:x+w]
            #char = plate_img[y:y+h,x:x+w]
            pro_process_char = self.pre_process_char(char)
            chars_img.append(pro_process_char)
            cv2.imwrite('static/images/temp/charsegment/字符{0}-{1}.jpg'.format(ran,i),pro_process_char)
            i += 1
        return chars_img

    def pre_process_char(self, char_img):
        h = char_img.shape[0]
        w = char_img.shape[1]
        char_size = self.char_size
        transform_char = np.eye(2,3,dtype=np.float32)
        max_w_h = w
        if h > w:
            max_w_h = h
        transform_char[0][2] = (max_w_h - w) / 2.0
        transform_char[1][2] = (max_w_h - h) / 2.0
        warp_image = cv2.warpAffine(char_img, transform_char, (max_w_h,max_w_h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
        out = cv2.resize(warp_image, (char_size,char_size))
        return out

    def rebuild_rect(self, sorted_rect, spec_indx, chinese_rect):
        #重建字符顺序
        out = [chinese_rect]
        count = 6
        for i in range(len(sorted_rect)):
            if i < spec_indx :
                continue
            out.append(sorted_rect[i])
            count -= 1
            if count == 0:
                break
        return out

    def get_chinese_rect(self, rect_spec):
        height = rect_spec[3]
        new_width = rect_spec[2] * 1.5
        x = rect_spec[0]
        y = rect_spec[1]
        newx = x - int(new_width)
        if newx < 0 :
            newx = 0
        chinese_rect = (newx,y,int(new_width), height)
        return chinese_rect

    def clear_rivet(self, plate_src):
        '''
        去除车牌上方的铆钉
        :param plate_src:
        :return:
        '''
        size = self.rivet_size
        row = plate_src.shape[0]
        col = plate_src.shape[1]
        jump = np.zeros((row))
        for i in range(row):
            jump_count = 0
            for j in range(col -1):
                if plate_src[i][j] != plate_src[i][j + 1]:
                    jump_count += 1
            jump[i] = float(jump_count)

        for i in range(row):
            if jump[i] < size :
                for j in range(col):
                    plate_src[i][j] = 0
        return plate_src

    def verify_size(self, char_img):
        ratio = 45.0 / 90.0
        row = char_img.shape[0]
        col = char_img.shape[1]
        if row == 0 or col == 0 :
            return False
        char_ratio = col * 1.0 / row
        error = 0.7
        min_height = 10.0
        max_height = 35.0
        min_ratio = 0.05
        max_ratio = ratio + ratio * error
        area = cv2.countNonZero(char_img)
        char_area = row * col
        perc_pixel = area * 1.0 / char_area
        return perc_pixel <= 1 and char_ratio > min_ratio and char_ratio < max_ratio and row > min_height and row < max_height

    def get_speci_rect(self, char_rects):
        #寻找城市代表字母

        max_height = 0
        max_width = 0
        pos = []
        #最大宽度和最大高度,最宽的为汉子，字母大概为汉字的0.8宽左右，高为0.9左右，且为于车牌的1/7到2/7之间
        for rect in char_rects:
            pos.append(rect[0])
            if rect[2] > max_width:
                max_width = rect[2]
            if rect[3] > max_height:
                max_height = rect[3]
        spec_indx = 0
        i = 0
        #(rect (x,y,w,h)）
        for rect in char_rects:
            midx = rect[0] + rect[2] / 2.0
            if (rect[2] > max_width * 0.7 or rect[3] > max_height * 0.9) and (midx < self.mat_width * 2.0 / 7 and midx > self.mat_width / 7.0):
                spec_indx = i
            i += 1
        return spec_indx