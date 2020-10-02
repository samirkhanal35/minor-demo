import os
from flask import Flask, flash, request, redirect, url_for, render_template, send_file
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import segment as sg


app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'


app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER




@app.route('/')
def minor():
    return render_template("minor.html")

@app.route('/minor_demo')
def minor_demo():
    img = cv2.imread('static/uploads/'+'minor_sample.jpg')
    h,w,d = img.shape
    # gray conversion
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Gaussian blurring to remove noise
    img2 = cv2.GaussianBlur(img1, (9, 9), 0)
    # binarization
    ret, img3 = cv2.threshold(img2,128,255,cv2.THRESH_BINARY)
    img4 = cv2.cvtColor(img3, cv2.COLOR_GRAY2BGR)
    c=sg.segFun(h,w,img4)
    # print(c)
    return render_template('minor.html', output=c)


if __name__ == '__main__':
    app.run(debug=True)
