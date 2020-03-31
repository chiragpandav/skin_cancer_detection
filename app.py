# Flask
from flask import Flask , request, render_template,  jsonify
from tensorflow.keras.models import load_model
import cv2

# Some utilites
import numpy as np
from util import base64_to_pil


# Declare a flask app
app = Flask(__name__)

print('Model loaded. Check http://127.0.0.1:5002/')

#add your model_path
model_path = '/home/chirag/ML_Model/skin.h5'

# Load your own trained model
model = load_model(model_path)
model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')

def model_predict(img, model):

    collecter = cv2.cvtColor(np.float32(img), cv2.COLOR_BGR2GRAY)

    resize_img = cv2.resize(collecter, (128, 128))
    final_img = np.array(resize_img)

    final_img = final_img.astype('float32')
    final_img /= 255

    extend_imgDims = np.expand_dims(final_img, axis=2)
    test_img = np.expand_dims(extend_imgDims, axis=0)

    index_id = model.predict(test_img)

    lable = np.argmax(index_id)

    return lable


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Make prediction
        preds = model_predict(img, model)

        if preds==0:
            result="MEL"
        elif preds==1:
            result="NV"
        elif preds==2:
            result="BCC"
        elif preds == 3:
            result = "AK"
        elif preds==4:
            result="BKL"
        elif preds==5:
            result="DF"
        elif preds ==6:
            result = "VASC"
        elif preds ==7:
            result = "SCC"
        else :
            result = "UNK"

        return jsonify(result=result, probability=int(preds))

    return None


if __name__ == '__main__':
    app.run(port=5002, threaded=False)
