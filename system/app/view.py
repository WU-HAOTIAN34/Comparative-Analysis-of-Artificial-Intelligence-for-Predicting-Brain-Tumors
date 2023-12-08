from keras import backend as K
from flask import render_template, request
from PIL import Image
import numpy as np
from keras.models import load_model
from app import app
import os


base_dir = os.path.dirname(os.path.abspath(__file__))

ARMNet = load_model(os.path.join(base_dir, '../model', 'ARMNet.h5'))
VGG16 = load_model(os.path.join(base_dir, '../model', 'VGG16.h5'))
ResNet50 = load_model(os.path.join(base_dir, '../model', 'ResNet50.h5'))
Xception = load_model(os.path.join(base_dir, '../model', 'Xception.h5'))


def preprocess_image(file):
    img = Image.open(file)
    img = img.resize((200, 200))
    print(os.path.join(base_dir, 'static', 'test_image'))
    img.save(os.path.join(base_dir, 'static', 'test_image', file.filename))
    print(os.path.join(base_dir, 'static', 'test_image'))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files["file"]

        if file:
            image = preprocess_image(file)

            model_choice = request.form.get("model_choice")

            if model_choice == "VGG16":
                model = VGG16
            elif model_choice == "Xception":
                model = Xception
            elif model_choice == "ResNet50":
                model = ResNet50
            elif model_choice == "ARMNet":
                model = ARMNet
            else:
                model = VGG16

            prediction = model.predict(image)
            class_index = np.argmax(prediction)
            classes = ["Glioma", "Meningioma", "Pituitary"]

            result = classes[class_index]

            return render_template("result.html", result=result, image_file='../static/test_image/' + file.filename)

    return render_template("index.html")
