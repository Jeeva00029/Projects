import re
import numpy as np
import os
from flask import Flask, app,request,render_template
from tensorflow.keras import models
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.python.ops.gen_array_ops import concat
#Loading the model
model=load_model(r"garbage.h5")

app=Flask(__name__)


# home page
@app.route('/')
def index():
    return render_template('index.html')

# prediction page
# user upload a image for garbage classification
@app.route('/prediction.html')
def prediction():
    return render_template('prediction.html')


# result page
@app.route('/result',methods=["GET","POST"])
def res():
    if request.method=="POST":
        f=request.files['image']
        basepath=os.path.dirname(__file__) 
        filepath=os.path.join(basepath,'uploads',f.filename)
        f.save(filepath)

        img=image.load_img(filepath,target_size=(128,128))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        prediction=np.argmax(model.predict(x), axis =1) 
        index=["cardboard","glass","metal","paper","plastic","trash"]
        result=str(index[prediction[0]])
        result
        return render_template('prediction.html',prediction=result)
        



""" Running our application """
if __name__ == "__main__":
    app.run(debug=True,port=8080)