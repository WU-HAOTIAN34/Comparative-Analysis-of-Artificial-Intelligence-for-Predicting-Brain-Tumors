from keras import backend as K
from flask import render_template, request
from PIL import Image
import numpy as np
from keras.models import load_model
import cv2
from app import app
import os
import matplotlib.cm as cm
import tensorflow as tf
import numpy as np
import keras
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing import image



base_dir = os.path.dirname(os.path.abspath(__file__))


def crop_image(image, image2):
    y_min = np.argmax(np.sum(image, axis=1)/image.shape[0] > 10)
    y_max = image.shape[0] - np.argmax(np.flipud(np.sum(image, axis=1)/image.shape[0] > 10))
    x_min = np.argmax(np.sum(image, axis=0)/image.shape[1] > 10)
    x_max = image.shape[1] - np.argmax(np.flipud(np.sum(image, axis=0)/image.shape[1] > 10))
    return image2[y_min:y_max, x_min:x_max, :]


def preprocess_image(file, size):
    img = Image.open(file)
    img.save(os.path.join(base_dir, 'static', 'test_image', file.filename))
    file = cv2.imread(os.path.join(base_dir, 'static', 'test_image', file.filename))
    image = cv2.cvtColor(file, cv2.COLOR_BGR2GRAY)
    image = crop_image(image, file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.bilateralFilter(image, 2, 50, 50)
    image = cv2.applyColorMap(image, cv2.COLORMAP_BONE)
    image = cv2.resize(image, (size, size))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":

        file = request.files["file"]

        if file:

            image = preprocess_image(file, 200)
            model_choice = request.form.get("model_choice")

            if model_choice == "VGG16":
                VGG16 = load_model(os.path.join(base_dir, '../model', 'VGG16.h5'))
                model = VGG16
                sove_plots('/root/autodl-tmp/system/app/static/test_image/' + file.filename, (200, 200), model, 'block5_conv3', 12)
                prediction = model.predict(image)
                VGG16 = None
            elif model_choice == "Xception":
                Xception = load_model(os.path.join(base_dir, '../model', 'Xception.h5'))
                model = Xception
                sove_plots('/root/autodl-tmp/system/app/static/test_image/' + file.filename, (200, 200), model, 'block14_sepconv2', 7)
                prediction = model.predict(image)
                Xception = None
            elif model_choice == "ResNet50":
                ResNet50 = load_model(os.path.join(base_dir, '../model', 'ResNet50.h5'))
                model = ResNet50
                sove_plots('/root/autodl-tmp/system/app/static/test_image/' + file.filename, (200, 200), model, 'conv5_block3_3_conv', 7)
                prediction = model.predict(image)
                ResNet50 = None
            elif model_choice == "ARMNet":
                ARMNet = load_model(os.path.join(base_dir, '../model', 'ARMNet.h5'))
                model = ARMNet
                sove_plots('/root/autodl-tmp/system/app/static/test_image/' + file.filename, (200, 200), model, 'add_3', 12)
                prediction = model.predict(image)
                ARMNet = None
            elif model_choice == "InceptionV3":
                InceptionV3 = load_model(os.path.join(base_dir, '../model', 'InceptionV3.h5'))
                model = InceptionV3
                sove_plots('/root/autodl-tmp/system/app/static/test_image/' + file.filename, (200, 200), model, 'conv2d_375', 4)
                prediction = model.predict(image)
                InceptionV3 = None
            elif model_choice == "MobileNetV2":
                MobileNetV2 = load_model(os.path.join(base_dir, '../model', 'MobileNetV2.h5'))
                model = MobileNetV2
                sove_plots('/root/autodl-tmp/system/app/static/test_image/' + file.filename, (200, 200), model, 'Conv_1', 7)
                prediction = model.predict(image)
                MobileNetV2 = None
            elif model_choice == "DenseNet121":
                DenseNet121 = load_model(os.path.join(base_dir, '../model', 'DenseNet121.h5'))
                model = DenseNet121
                sove_plots('/root/autodl-tmp/system/app/static/test_image/' + file.filename, (200, 200), model, 'conv5_block16_2_conv', 6)
                prediction = model.predict(image)
                DenseNet121 = None
            elif model_choice == "SqueezeNet":
                SqueezeNet = load_model(os.path.join(base_dir, '../model', 'SqueezeNet.h5'))
                model = SqueezeNet
                sove_plots('/root/autodl-tmp/system/app/static/test_image/' + file.filename, (200, 200), model, 'fire9/concat', 11)
                prediction = model.predict(image)
                SqueezeNet = None
            elif model_choice == "EfficientNetB0":
                EfficientNetB0 = load_model(os.path.join(base_dir, '../model', 'EfficientNetB0.h5'))
                model = EfficientNetB0
                sove_plots('/root/autodl-tmp/system/app/static/test_image/' + file.filename, (200, 200), model, 'top_conv', 7)
                prediction = model.predict(image)
                EfficientNetB0 = None
            elif model_choice == "MFLNet":
                MFLNet = load_model(os.path.join(base_dir, '../model', 'MFLNet.h5'))
                model = MFLNet
                sove_plots('/root/autodl-tmp/system/app/static/test_image/' + file.filename, (200, 200), model, 'conv2d_27', 12)
                prediction = model.predict(image)
                MFLNet = None
            elif model_choice == "AlexNet":
                AllexNet = load_model(os.path.join(base_dir, '../model', 'AlexNet.h5'))
                model = AllexNet
                sove_plots('/root/autodl-tmp/system/app/static/test_image/' + file.filename, (200, 200), model, 'Conv2D-5', 11)
                prediction = model.predict(image)
                AllexNet = None
            elif model_choice == "CapsuleNet":
                CapsuleNet = load_model(os.path.join(base_dir, '../model', 'MFLNet.h5'))
                model = CapsuleNet
                sove_plots('/root/autodl-tmp/system/app/static/test_image/' + file.filename, (200, 200), model, 'conv2d_28', 12)
                prediction = model.predict(image)
                CapsuleNet = None
            elif model_choice == "DNN+SVM":
                image = preprocess_image(file, 241)
                DNN_SVM = load_model(os.path.join(base_dir, '../model', 'DNNwithSVM.h5'))
                model = DNN_SVM
                sove_plots('/root/autodl-tmp/system/app/static/test_image/' + file.filename, (241, 241), model, 'max_pooling2d_2', 6)
                DNN_SVM = model.predict(image)
                AllexNet = None
            elif model_choice == "NeurolNet19":
                image = preprocess_image(file, 128)
                NeuroNet19 = load_model(os.path.join(base_dir, '../model', 'NeuroNet.h5'))
                model = NeuroNet19
                sove_plots('/root/autodl-tmp/system/app/static/test_image/' + file.filename, (128, 128), model, 'conv2d_3', 4)
                prediction = model.predict(image)
                NeuroNet19 = None
            elif model_choice == "WeightedEM":
                WeightedEM = load_model(os.path.join(base_dir, '../model', 'WeightedEM.h5'))
                model = WeightedEM
                sove_plots('/root/autodl-tmp/system/app/static/test_image/' + file.filename, (224, 224), model, 'block5_pool', 7)
                prediction = model.predict(image)
                WeightedEM = None
            elif model_choice == "proposed":
                proposed = load_model(os.path.join(base_dir, '../model', 'proposed.h5'))
                model = proposed
                sove_plots('/root/autodl-tmp/system/app/static/test_image/' + file.filename, (200, 200), model, 'multiply_2', 12)
                prediction = model.predict(image)
                proposed = None
            elif model_choice == "CDANet":
                CDANet = load_model(os.path.join(base_dir, '../model', 'CDANet.h5'))
                model = CDANet
                sove_plots('/root/autodl-tmp/system/app/static/test_image/' + file.filename, (200, 200), model, 'conv5_block16_2_conv', 6)
                prediction = model.predict(image)
                CDANet = None
            else:
                DenseNet121 = load_model(os.path.join(base_dir, '../model', 'DenseNet121.h5'))
                model = DenseNet121
                sove_plots('/root/autodl-tmp/system/app/static/test_image/' + file.filename, (200, 200), model, 'conv5_block16_2_conv', 6)
                prediction = model.predict(image)
                DenseNet121 = None
            

            class_index = np.argmax(prediction)
            classes = ["Glioma", "Meningioma", "Pituitary"]

            result = classes[class_index]

            return render_template("result.html", result=result, image_file='../static/test_image/' + file.filename)

    return render_template("index.html")


def display_gradcam(img, heatmap, cam_path="cam.jpg", alpha=0.4,preds=[0,0,0,0], plot=None):

    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")

    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = image.array_to_img(superimposed_img)

    plot.imshow(superimposed_img)
    plot.axis('off')


def gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
   
    grad_model = tf.keras.models.Model(
        [model.input], [model.get_layer(last_conv_layer_name).output,  model.output]
    )
    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
        
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    
    last_conv_layer_output = last_conv_layer_output
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def evaluate_model(model, data, labels):
  false_predictions = []
  for img, label in zip(data, labels):
    image = image = tf.expand_dims(img, axis=0)
    predicted = model.predict(image)
    true_label= np.argmax(label)
    predicted_label = np.argmax(predicted)
    confidence = predicted[0][predicted_label]

    if true_label != predicted_label:
      false_predictions.append([img, true_label, predicted_label, confidence, predicted])
  return false_predictions


def evaluate_model(model, data, labels):
  true_predictions = []
  for img, label in zip(data, labels):
    image = tf.expand_dims(img, axis=0)
    predicted = model.predict(image)
    true_label= np.argmax(label)
    predicted_label = np.argmax(predicted)
    confidence = predicted[0][predicted_label]

    if true_label == predicted_label:
      true_predictions.append([img, true_label, predicted_label, confidence, predicted])
  return true_predictions


def get_img_array(img_path, size):
    
    img = image.load_img(img_path, target_size=size)
    array = image.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array


def sove_plots(path, size, model, last_conv_layer_name, size2):
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(20, 10))
    plt.subplots_adjust(bottom=0)
    img_array = get_img_array(path,size)
    img= cv2.imread(path)
    img = cv2.resize(img, size)
    model.layers[-1].activation = None
    preds = model.predict(img_array)
    
    i = np.argmax(preds[0])
    print(i)
    if i==0:
        actual = "glioma"
        predicted_label_class = "glioma"
    elif i==1:
        actual = "meningioma"
        predicted_label_class = "meningioma"
    elif i==2:
        actual = "pituitary tumor"
        predicted_label_class = "pituitary tumor"
    
    title = "Predicted Label: {} \n".format(
            predicted_label_class)
    plt.axis('off')
    heatmap = gradcam_heatmap(img_array, model, last_conv_layer_name)
    heatmap = np.reshape(heatmap, (size2,size2)) 
    display_gradcam(img, heatmap, preds=preds[0], plot=ax1)
    _ = ax2.imshow(img)
    _ = ax3.imshow(heatmap)
    plt.savefig(path, bbox_inches='tight')
    plt.show()
    plt.close()
    print('------------')