#!/usr/bin/env python
#coding=gbk
#�������ID,����һ��ʶ���ΨһID,Ŀ���Ƕ�ͼƬ�����������������ǿ�������ÿ�ζ��ӷ���������
import random
class ID(object):
    ID = 0
    @classmethod
    def get_ID(cls,is_new = False):
        if is_new :
            cls.ID = random.randint(0,10000)
        return cls.ID