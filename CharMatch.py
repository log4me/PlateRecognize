#!/usr/bin/env python
#coding=gbk


import cv2
import numpy as np


#图像相似度计算
# 灰度直方图实现
def classify_gray_hist(image1, image2, size=(256, 256)):
    # 先计算直方图
    # bins 取为16
    image1 = cv2.resize(image1, size)
    image2 = cv2.resize(image2, size)
    hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])

    # 计算直方图的重合度
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    degree = degree / len(hist1)
    return degree


# 单通道直方图的相似值
def calculate(image1, image2):
    hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
    # 计算直方图的重合度
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    degree = degree / len(hist1)
    return degree


# 通过得到每个通道的直方图来计算相似度
def classify_hist_with_split(image1, image2, size=(256, 256)):
    # 将图像resize后，分离为三个通道，再计算每个通道的相似值
    image1 = cv2.resize(image1, size)
    image2 = cv2.resize(image2, size)
    sub_image1 = cv2.split(image1)
    sub_image2 = cv2.split(image2)
    sub_data = 0
    for im1, im2 in zip(sub_image1, sub_image2):
        sub_data += calculate(im1, im2)
    sub_data = sub_data / 3
    return sub_data


# 平均哈希算法计算
def classify_aHash(image1, image2):
    image1 = cv2.resize(image1, (8, 8))
    image2 = cv2.resize(image2, (8, 8))
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    hash1 = getHash(gray1)
    hash2 = getHash(gray2)
    return Hamming_distance(hash1, hash2)


def classify_pHash(image1, image2):
    image1 = cv2.resize(image1, (32, 32))
    image2 = cv2.resize(image2, (32, 32))
    gray1 = image1
    gray2 = image2
    if len(image1.shape) >= 3:
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    if len(image2.shape) >= 3:
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # 将灰度图转为浮点型，再进行dct变换
    dct1 = cv2.dct(np.float32(gray1))
    dct2 = cv2.dct(np.float32(gray2))
    # 取左上角的8*8，这些代表图片的最低频率
    # 这个操作等价于c++中利用opencv实现的掩码操作
    # 在python中进行掩码操作，可以直接这样取出图像矩阵的某一部分
    dct1_roi = dct1[0:8, 0:8]
    dct2_roi = dct2[0:8, 0:8]
    hash1 = getHash(dct1_roi)
    hash2 = getHash(dct2_roi)
    return Hamming_distance(hash1, hash2)


# 输入灰度图，返回hash
def getHash(image):
    avreage = np.mean(image)
    hash = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] > avreage:
                hash.append(1)
            else:
                hash.append(0)
    return hash


# 计算汉明距离
def Hamming_distance(hash1, hash2):
    num = 0
    for index in range(len(hash1)):
        if hash1[index] != hash2[index]:
            num += 1
    return num


if __name__ == '__main__':
    img1 = cv2.imread('res/model/image/8.bmp')
    cv2.imshow('img1', img1)
    img2 = cv2.imread('static/images/temp/charsegment/字符1-4.jpg')
    img3 = cv2.imread('static/images/temp/charsegment/字符3-5.jpg')
    cv2.imshow('img2', img2)
    degree0 = classify_gray_hist(img1, img2)
    degree1 = classify_hist_with_split(img1,img2)
    degree2 = classify_aHash(img1,img2)
    degree3 = classify_pHash(img1,img2)
    degree11 = classify_gray_hist(img1, img3)
    degree11 = classify_hist_with_split(img1, img3)
    degree21 = classify_aHash(img1, img3)
    degree31 = classify_pHash(img1, img3)
    print degree0
    cv2.waitKey(0)