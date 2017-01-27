# PlateRecognize
A plate recognize program written by python and opencv.

# Plate Recognize

## 简要说明
此程序采用`python27`+`opencv`编写，但是**安装包可以运行于64位windows 7以及更高版本而不依赖系统`python`和`opencv`。**

## 简介

**plate recognize**是一个用于车牌识别的python程序。

## 依赖
- `python2.7` 
- `opencv_python(3.0以下，2.4以上)` 
- `numpy`- `flask`

## 模块
- `Function.py` : 特征提取，用于svm特征和ann特征- `Platelocate.py` : 车牌定位，用于定位图中所有可能车牌，包含图片的基本处理 
- `plateJudge.py` : 车牌判断，用SVM判断定位后的矩形是否是车牌
- `plateFeatures.py` : 图片基本特征(直方图特征)- `PlateDetect.py` : 用于车牌检测，包装了1和2的功能- `charSegment.py` : 字符分割，分割出车牌中的字符- `CharIdentify.py` : 字符识别，识别出分割的字符
    - 此模块实现了字符识别功能，含有两种方法。
        1. 神经网络识别，训练数据太少，识别率不高。
        2. 使用相似度匹配的方法。
        3. 此模块**默认采用相似度匹配**算法，可选使用ANN。
- `CharRecognize.py` : 用于字符识别，包装了6和7的功能
- `PlateRecognize.py` : 车牌字符检测，包装了5和8的功能- `app.py` : GUI界面(web实现)
- `Util.py` : 辅助功能，获取唯一ID,用于标识图片(防止浏览器从缓存加载)
- `TrainANN.py`:用于训练ANN 

## 文件夹
- `res` : 资源文件夹，包含`SVM`，`ANN`模型，以及模板文件
- `static`: GUI(WEB)资源文件夹
- `static/images/temp`:处理缓存文件夹
- `static/images/temp/charsegment`:车牌分割产生的字符图片
- `static/images/temp/plate`:分割产生的车牌文件
- `static/images/temp/platelocate`:图片预处理文件夹
- `test_img`:测试图片文件夹

## 处理流程


![](http://i1.piimg.com/4851/e01aa4bec7aceb26.png)






