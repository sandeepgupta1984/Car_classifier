#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Sandeep
"""
from fastai import *
from fastai.vision import *
import PIL.Image
import numpy as np
from flasgger import Swagger

from flask import Flask, request
app = Flask(__name__)
swagger = Swagger(app)

path_cars = Path("./src_cars")

classes = ['i10','i20']
empty_data = ImageDataBunch.load_empty(path_cars)
learn = create_cnn(empty_data1, models.resnet34, pretrained=False).load('ImgItemList')

@app.route('/predict_car', methods=['POST'])
def predict_car():
    """Example endpoint returning a prediction of car
    ---
    parameters:
        - name: image
          in: formData
          type: file
          required: true
    responses:
        200:
            description: "i10/i20 classification"
    """
    img = open_image(request.files['image'])
    pred_class, pred_idx, outputs = learn.predict(img)
    return str(pred_class)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7000)