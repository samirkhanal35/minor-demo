import os
from flask import Flask, flash, request, redirect, url_for, render_template, send_file
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import gary_convert as gc
import noise_red as nr
import binarize as br
# import gaussfun as gau
import segment as sg


app = Flask(__name__)
# img =''
# global filename, current_file
# filename = ''
# current_file =''
# basedir = os.path.abspath(os.path.dirname(__file__))

UPLOAD_FOLDER = 'static/uploads'


app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER




@app.route('/', methods=['POST','GET'])
def minor():
    return render_template("minor.html")

@app.route('/major', methods=['POST','GET'])
def major():
    return render_template("major.html")

@app.route('/minor_demo')
def minor_demo():
    img = cv2.imread('static/uploads/'+'minor_sample.jpg')
    h,w,d = img.shape
    gc.gray_con(h,w,img)
    # print("Grayed")
    nr.noisered(h,w,img)
    # print("filtered")
    #gau.gauss(heighty,widthx,img)
    # print("Blurred")
    br.bin_(h,w,img)
    # print("Binarized")
    c=sg.segFun(h,w,img)
    print(c)
    return render_template('minor.html', output=c)

@app.route('/major_demo')
def major_demo():
    return render_template('major.html')



if __name__ == '__main__':
    app.run(debug=True)
