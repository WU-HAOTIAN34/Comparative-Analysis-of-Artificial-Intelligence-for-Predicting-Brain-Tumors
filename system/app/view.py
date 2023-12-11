from keras import backend as K
from flask import render_template, request
from PIL import Image
import numpy as np
from keras.models import load_model
import cv2
from app import app
import os


base_dir = os.path.dirname(os.path.abspath(__file__))


def crop_image(image, image2):
    y_min = np.argmax(np.sum(image, axis=1)/image.shape[0] > 10)
    y_max = image.shape[0] - np.argmax(np.flipud(np.sum(image, axis=1)/image.shape[0] > 10))
    x_min = np.argmax(np.sum(image, axis=0)/image.shape[1] > 10)
    x_max = image.shape[1] - np.argmax(np.flipud(np.sum(image, axis=0)/image.shape[1] > 10))
    return image2[y_min:y_max, x_min:x_max, :]


def preprocess_image(file):
    img = Image.open(file)
    img.save(os.path.join(base_dir, 'static', 'test_image', file.filename))
    file = cv2.imread(os.path.join(base_dir, 'static', 'test_image', file.filename))
    image = cv2.cvtColor(file, cv2.COLOR_BGR2GRAY)
    image = crop_image(image, file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.bilateralFilter(image, 2, 50, 50)
    image = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
    image = cv2.resize(image, (200, 200))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":

        file = request.files["file"]

        if file:

            image = preprocess_image(file)
            model_choice = request.form.get("model_choice")

            if model_choice == "VGG16":
                VGG16 = load_model(os.path.join(base_dir, '../model', 'VGG16.h5'))
                model = VGG16
                prediction = model.predict(image)
                VGG16 = None
            elif model_choice == "Xception":
                Xception = load_model(os.path.join(base_dir, '../model', 'Xception.h5'))
                model = Xception
                prediction = model.predict(image)
                Xception = None
            elif model_choice == "ResNet50":
                ResNet50 = load_model(os.path.join(base_dir, '../model', 'ResNet50.h5'))
                model = ResNet50
                prediction = model.predict(image)
                ResNet50 = None
            elif model_choice == "ARMNet":
                ARMNet = load_model(os.path.join(base_dir, '../model', 'ARMNet.h5'))
                model = ARMNet
                prediction = model.predict(image)
                ARMNet = None
            elif model_choice == "InceptionV3":
                InceptionV3 = load_model(os.path.join(base_dir, '../model', 'InceptionV3.h5'))
                model = InceptionV3
                prediction = model.predict(image)
                InceptionV3 = None
            elif model_choice == "MobileNetV2":
                MobileNetV2 = load_model(os.path.join(base_dir, '../model', 'MobileNetV2.h5'))
                model = MobileNetV2
                prediction = model.predict(image)
                MobileNetV2 = None
            elif model_choice == "DenseNet121":
                DenseNet121 = load_model(os.path.join(base_dir, '../model', 'DenseNet121.h5'))
                model = DenseNet121
                prediction = model.predict(image)
                DenseNet121 = None
            elif model_choice == "SqueezeNet":
                SqueezeNet = load_model(os.path.join(base_dir, '../model', 'SqueezeNet.h5'))
                model = SqueezeNet
                prediction = model.predict(image)
                SqueezeNet = None
            elif model_choice == "EfficientNetB0":
                EfficientNetB0 = load_model(os.path.join(base_dir, '../model', 'EfficientNetB0.h5'))
                model = EfficientNetB0
                prediction = model.predict(image)
                EfficientNetB0 = None
            elif model_choice == "MFLNet":
                MFLNet = load_model(os.path.join(base_dir, '../model', 'MFLNet.h5'))
                model = MFLNet
                prediction = model.predict(image)
                MFLNet = None
            elif model_choice == "AlexNet":
                AllexNet = load_model(os.path.join(base_dir, '../model', 'AlexNet.h5'))
                model = AllexNet
                prediction = model.predict(image)
                AllexNet = None
            elif model_choice == "test":
                test = load_model(os.path.join(base_dir, '../model', 'test.h5'))
                model = test
                prediction = model.predict(image)
                AllexNet = None
            else:
                DenseNet121 = load_model(os.path.join(base_dir, '../model', 'DenseNet121.h5'))
                model = DenseNet121
                prediction = model.predict(image)
                DenseNet121 = None

            class_index = np.argmax(prediction)
            classes = ["Glioma", "Meningioma", "Pituitary"]

            result = classes[class_index]

            return render_template("result.html", result=result, image_file='../static/test_image/' + file.filename)

    return render_template("index.html")
