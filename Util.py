#!/usr/bin/env python
#coding=gbk
#随机生成ID,代表一次识别的唯一ID,目的是对图片的命名产生随机数，强制浏览器每次都从服务器加载
import random
class ID(object):
    ID = 0
    @classmethod
    def get_ID(cls,is_new = False):
        if is_new :
            cls.ID = random.randint(0,10000)
        return cls.ID