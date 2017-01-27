#!/usr/bin/env python
#coding=gbk
import os
import cv2
import numpy as np
import Function
import random
ann = cv2.ANN_MLP(np.array([140,65]),cv2.ANN_MLP_SIGMOID_SYM,1,1)
ann.load("res/model/ann.xml")

label = np.array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
         0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
                    ]], dtype=np.float32)
#label[0][1] = 1
#feature = feature.reshape(1,-1)
weight = np.ndarray((1,1),np.float32)
weight[0][0] = 1
#ann.train(feature,label,weight)
#ann.save('ann.xml')
prefix_num = r'res/image/chars2/'
prefix_ch = r'res/image/charsChinese/'
nums = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J',
                       'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
chinese = ["zh_cuan", "zh_e", "zh_gan", "zh_gan1",
                    "zh_gui", "zh_gui1", "zh_hei", "zh_hu", "zh_ji", "zh_jin",
                    "zh_jing", "zh_jl", "zh_liao", "zh_lu", "zh_meng",
                    "zh_min", "zh_ning", "zh_qing", "zh_qiong", "zh_shan",
                    "zh_su", "zh_sx", "zh_wan", "zh_xiang", "zh_xin", "zh_yu",
                    "zh_yu1", "zh_yue", "zh_yun", "zh_zang", "zh_zhe"]
for i in range(1):
    for i in range(len(nums)):
        dir = prefix_num + nums[i]
        label[0] = 0
        label[0][i] = 1
        files = []
        try:
            files = os.listdir(dir)
        except:
            print('Error')
        j = 0

        for file in files:
            img = cv2.imread(dir + '/' + file)
            if  img is None or len(img) == 0:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
            feature = Function.Func.features(img,10)
            feature = feature.reshape(1,-1)
            ann.train(feature,label,weight)
            out = ann.predict(feature)
            j += 1
            if j % 100 == 0:
                ann.save('res/model/ann.xml')
                print('save')

            print(dir + '/' + file)
        ann.save('res/model/ann.xml')
        print('end:{}'.format(i))
ann.save('res/model/ann.xml')

