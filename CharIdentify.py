#!/usr/bin/env python
#coding=gbk
import cv2
import numpy
import Function
import numpy as np
import os
import CharMatch
class CharIdentify(object):
    def __init__(self):
        self.num_chinese = 31
        self.chinese_chars = self.getChineseDict()
        self.ann = cv2.ANN_MLP()
        self.ann.clear()
        self.ann.load('res/model/ann.xml')
        self.models_prefix = 'res/model/image/'
        self.num_character = 34
        self.num_all = self.num_character + self.num_chinese
        self.str_characters = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','U','V','W','X','Y','Z',
                               "zh_cuan", "zh_e", "zh_gan", "zh_gan1",
                               "zh_gui", "zh_gui1", "zh_hei", "zh_hu", "zh_ji", "zh_jin",
                               "zh_jing", "zh_jl", "zh_liao", "zh_lu", "zh_meng",
                               "zh_min", "zh_ning", "zh_qing", "zh_qiong", "zh_shan",
                               "zh_su", "zh_sx", "zh_wan", "zh_xiang", "zh_xin", "zh_yu",
                               "zh_yu1", "zh_yue", "zh_yun", "zh_zang", "zh_zhe"
                               ]
        self.models = self.get_model()
        self.predict_size = 10
    def get_model(self):
        models = []
        for i in range(len(self.str_characters)):
            dir = self.models_prefix + self.str_characters[i]
            files = []
            try:
                files = os.listdir(dir)
            except:
                print('Error')
            j = 0
            current_num_models = []
            for file in files:
                img = cv2.imread(dir + '/' + file)
                if img is None or len(img) == 0:
                    continue
                else:
                    current_num_models.append(img)
            models.append(current_num_models)
        return models

    def getChineseDict(self):
        return {
            "zh_cuan": "川",
            "zh_e": "鄂",
            "zh_gan": "赣",
            "zh_gan1": "甘",
            "zh_gui": "贵",
            "zh_gui1": "桂",
            "zh_hei": "黑",
            "zh_hu": "沪",
            "zh_ji": "冀",
            "zh_jin": "津",
            "zh_jing": "京",
            "zh_jl": "吉",
            "zh_liao": "辽",
            "zh_lu": "鲁",
            "zh_meng": "蒙",
            "zh_min": "闽",
            "zh_ning": "宁",
            "zh_qing": "青",
            "zh_qiong": "琼",
            "zh_shan": "陕",
            "zh_su": "苏",
            "zh_sx": "晋",
            "zh_wan": "皖",
            "zh_xiang": "湘",
            "zh_xin": "新",
            "zh_yu": "豫",
            "zh_yu1": "渝",
            "zh_yue": "粤",
            "zh_yun": "云",
            "zh_zang": "藏",
            "zh_zhe": "浙"
        }
    def char_match(self, img, is_chinese = False, is_spec = False):
        if is_chinese:
            min_degree = 0
            min_index = -1
            for i in range(34, 65):
                degree = 0.0
                j = 0
                for model in self.models[i]:
                    tmp = CharMatch.classify_pHash(img, model)
                    if j == 0:
                        degree = 1.0 * tmp
                    else:
                        if tmp < degree:
                            degree = tmp
                    j += 1

                if min_index == -1:
                    min_index = i
                    min_degree = degree
                else:
                    if degree < min_degree:
                        min_degree = degree
                        min_index = i
            return min_index
        else:
            min_degree = 0
            min_index = -1
            for i in range(0, 34):
                degree = 0.0
                j = 0
                for model in self.models[i]:
                    tmp = CharMatch.classify_pHash(img,model)
                    if j == 0:
                        degree = 1.0 * tmp
                    else:
                        if tmp < degree:
                            degree = tmp
                    j += 1

                if min_index == -1:
                    min_index = i
                    min_degree = degree
                else:
                    if degree < min_degree:
                        min_degree = degree
                        min_index = i
            return min_index

    def char_identify(self, img, is_chinese, is_spec):
        result = ''
        #ANN 没有足够数据训练，改成相似度匹配
        #f = Function.Func.features(img, 10)
        #index = self.classify(f, is_chinese,is_spec)
        #if is_chinese == False:
            #result = self.str_characters[index]
        #else:
            #key = self.str_chinese[index - self.num_character]
            #result = self.chinese_chars[key]
        #return result
        index = self.char_match(img,is_chinese,is_spec)
        if not is_chinese:
            return self.str_characters[index]
        else:
            key = self.str_characters[index]
            return self.chinese_chars[key]

    def classify(self, img, is_chinesse, is_spec):
        result = -1
        img = img.reshape(1,-1)
        img.dtype = 'float32'
        output = None
        _,output = self.ann.predict(img)
        ann_min = self.num_character
        #判断是否是中文
        if is_chinesse == False :
            if is_spec == True:
                ann_min = 10
            else :
                ann_min = 0
        else :
            ann_min = self.num_character
        ann_max = self.num_character
        if is_chinesse == True:
            ann_max = self.num_all
        max_val = -2
        out_float = output.copy()
        out_float.dtype = 'float32'
        #在ann_min和ann_max范围内找最大值（ann返回值为 one hot vector ,共有65个，把最大的作为结果
        for i in range(ann_min, ann_max):
            val = out_float[0][i]
            if val > max_val :
                max_val = val
                result = i
        return result
if __name__ == '__main__':
    char_i = CharIdentify()
