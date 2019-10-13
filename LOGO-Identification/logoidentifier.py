# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 14:11:45 2019

@author: HEMIL
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 20:42:21 2019

@author: HEMIL
"""

# Convolutional Neural Network

# Importing the Keras libraries and packages
import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.preprocessing import image

# Initialising the CNN
classifier = Sequential()

# Adding first convolutional layer
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
# pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(64, 3, 3, activation = 'relu'))
# Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening
classifier.add(Flatten())

# Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 6, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Fitting the CNN to the images
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')

classifier.fit_generator(training_set,
                         samples_per_epoch = 2900,
                         nb_epoch = 20,
                         validation_data = test_set,
                         nb_val_samples = 96)


classifier.summary()


# SAVEING THE MODEL
# Saving weights
fname = (r"C:\Users\HEMIL\Desktop\Logo detection\logo_classifier_weights-CNN.h5")
classifier.save(fname)
# Loading weights
fname = (r"C:\Users\HEMIL\Desktop\Logo detection\logo_classifier_weights-CNN.h5")
model = load_model(fname)
print("Model loaded")


"""Output is just for console"""
# Testing the model on new data
class_names = list(training_set.class_indices.keys())

test_img = image.load_img('downloadhonda.jpg', target_size = (64, 64))
test_img = image.img_to_array(test_img)
test_img = np.expand_dims(test_img,axis = 0)
result = model.predict(test_img)
result = result.flatten()
m = max(result)

for index, item in enumerate(result):
    if item == m:
        pred = class_names[index]
print(pred)      

"""
@README
ABOUT THE DATASET AND ALGORITHM
1) Made the algorithm using keras api as a basic model and not with tensorflow. Also used image preprocessing from 
    KERAS documentation. (flow from the directory) LINK : https://keras.io/preprocessing/image/
2) The task was to identify logos so I found a dataset containing logos in closeup form from the internet and used that.
    But we can always add more images of car logo from little far to the dataset to make it more accurate and versetile.
3) after that I saved the model in .h5 file and used it to identify the logos from the pictures downloaded from the internet on
    python console.
4) Created a Django Webapp and added the saved .h5 file to it to run on a local host server.
"""  


"""
@README

FINE TUNING THE MODEL

1) First I tried with only one convolution layer with 32 features (32,3,3) and input shape of image as (32,32,3) and total epochs 10
    train acc was 79% and test acc was 91%
2) second time I used 2 convolution and pooling layer with same 32 features (32,3,3) and input shape ((32,32,3) and total epochs 10
   train acc was 83% and test acc was 86%
3) at last I used 2 convolution layers one with 32 features (32,3,3) and second with 64 feature detector (64,3,3) and also 
    increse the size of the image from 32 pixel to 64 ie input shape = (64,64,3) and also incresed number of epoch to 20.
    train acc was 95 and test acc was also 95
    
4) We can further fine tune the model by increasing epochs or adding learning rate or adding more convolution layers with high 
    feature detector of size 128 and more. Also we can increase the size of the input image or add more data to the dataset
"""


      
                
