from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

model = load_model("model.h5")

classes = ['arborio', 'basmati', 'ipsala', 'jasmine', 'karacadag']

def predict(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    prediction = model.predict(img)
    return classes[np.argmax(prediction)]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dash.html')

@app.route('/testimonials')
def testimonials():
    return render_template('tes.html')

@app.route('/contact')
def contact():
    return render_template('cont.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_page():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filepath = os.path.join("static", file.filename)
            file.save(filepath)

            result = predict(filepath)

            return render_template('pred.html', result=result, image=filepath)

    return render_template('pred.html')


if __name__ == '__main__':
    app.run(debug=True)