from django.shortcuts import render
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
import numpy as np
global graph,model

#initializing the graph
graph = tf.get_default_graph()

#loading our trained model
print("Keras model loading.......")
model = load_model('logoidentify/logo_classifier_weights-CNN.h5')
print("Model loaded!!")



#creating a dictionary of classes
class_dict = {'AUDI': 0,
            'CHEVROLET': 1,
            'FERRARI': 2,
            'FORD': 3,
            'HONDA': 4,
            'TESLA': 5}

class_names = list(class_dict.keys())

def prediction(request):
    if request.method == 'POST' and request.FILES['myfile']:
        post = request.method == 'POST'
        myfile = request.FILES['myfile']
        img = image.load_img(myfile, target_size=(64, 64))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        with graph.as_default():
            preds = model.predict(img)
        preds = preds.flatten()
        m = max(preds)
        for index, item in enumerate(preds):
            if item == m:
                result = class_names[index]
        return render(request, "logoidentify/pred.html", {
            'result': result})
    else:
        return render(request, "logoidentify/pred.html")