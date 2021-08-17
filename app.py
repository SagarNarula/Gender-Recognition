from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='model.hdf5'

# Load your trained model
model = load_model(MODEL_PATH)

@app.route('/')
def show():
  return render_template('index.html')



def model_predict(img_path, model):
  img = image.load_img(img_path,target_size=(100,100))
  # Preprocessing the image
  x = image.img_to_array(img)
  # x = np.true_divide(x, 255)
  ## Scaling
  x=x/255
  x = np.expand_dims(x, axis=0)
  preds = model.predict(x)
  print(preds)
  #preds=np.argmax(preds, axis=1)
  if preds[0][0]==1:
    preds="Gender is Male"
  else:
     preds="Gender is Female"
    
    
    
  return preds

@app.route('/predict', methods=['GET', 'POST'])
def upload():
  if request.method == 'POST':
    f = request.files['file']
    #basepath = os.path.dirname(__file__)
    # Save the file to ./uploads
    file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
    f.save(file_path)

        # Make prediction
    preds = model_predict(file_path, model)
    result=preds
    return result
  return None


if __name__ == '__main__':
    app.run(debug=True)
