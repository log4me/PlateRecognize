#!/usr/bin/env python
#coding=gbk
from flask import Flask, request, session, g, redirect, url_for, render_template, flash, make_response
import os
import sys
from werkzeug.utils import secure_filename
from Util import ID
import PlateRecognize
import cv2
reload(sys)
sys.setdefaultencoding('gbk')
app = Flask(__name__)
plate_recognize = PlateRecognize.PlateRecognize()
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'images')
ALLOWED_EXTENSIONS=set(['jpg','png','JPG','PNG'],)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/recognize',methods=['POST','GET'])
def recognize():
    if request.method == 'POST':
        ID.get_ID(True)
        f = request.files['image']
        if f and allowed_file(f.filename):
            fname = secure_filename(f.filename)
            f.save(os.path.join(app.config['UPLOAD_FOLDER'],fname))
            full_name = 'static/images/' + f.filename
            out = plate_recognize.plate_recognize('static/images/'+fname)
            return render_template('show.html',  plates = out, count = len(out), ID=ID.get_ID())

        return fail_msg("不支持的图片格式!", url_for('/'))
    return render_template('index.html')

def fail_msg(content,return_url = None):
    if not return_url :
        return '<img src='+url_for('static',filename='image/f.png')+'><font size=6 color=red>'+content+'</font>'
    else :
        return '<meta http-equiv="refresh" content=1;url="'+return_url+'">' +'\n'+'<img src='+url_for('static',filename='image/f.png')+'><font size=6 color=red>'+content+'</font>'

if __name__ == '__main__':
    app.run(port=5000,debug=False)
