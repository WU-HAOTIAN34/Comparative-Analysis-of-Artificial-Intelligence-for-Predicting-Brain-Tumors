import matplotlib.cm as cm
import tensorflow as tf
import numpy as np
import keras
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing import image


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
   
  title = "True Label: {} \n Predicted Label: {} \n".format(
        actual, predicted_label_class)
  plt.axis('off')
  heatmap = gradcam_heatmap(img_array, model, last_conv_layer_name)
  heatmap = np.reshape(heatmap, (size2,size2)) 
  display_gradcam(img, heatmap, preds=preds[0], plot=ax1)
  _ = ax2.imshow(img)
  _ = ax3.imshow(heatmap)
  ax1.set_title("GradCam")
  ax2.set_title(title)
  ax3.set_title('Attention Map')
  plt.show()
  plt.close()
  print('------------')