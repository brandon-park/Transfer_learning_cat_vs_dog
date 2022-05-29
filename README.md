
# <center> Cats Vs Dogs Image Classification using Transfer Learning

<center> <img src="https://github.com/brandon-park/Transfer_learning_cat_vs_dog/blob/main/Header_Cats-vs-Dogs-951x512.png?raw=true" width="70%"/>

## TOC:

1. [Introduction](#Introduction)
2. [Data preprocessing](#preprocessing)
3. [Transfer Learning](#Transfer-Learning)
4. [Prediction on testing data](#Prediction)

## Introduction <a name="Introduction"></a>


### Benefits of Transfer Learning

In transfer learning, a machine learning model is trained on one kind of problem, and then used on a different but related problem, drawing on the knowledge it already has while learning its new task. This could be as simple as training a model to recognize giraffes in images, and then making use of this pre-existing expertise to teach the same model to recognize pictures of sheep.

The main benefits of transfer learning for machine learning include:

- Removing the need for a large set of labelled training data for every new model.
- Improving the efficiency of machine learning development and deployment for multiple models.
- A more generalised approach to machine problem solving, leveraging different algorithms to solve new challenges.
- Models can be trained within simulations instead of real-world environments.

https://www.seldon.io/transfer-learning#:~:text=The%20main%20benefits%20of%20transfer,model%20will%20be%20pre%2Dtrained
https://www.sparkcognition.com/transfer-learning-machine-learning/?utm_source=www.google.com&utm_medium=organic&utm_campaign=Google&referrer-analytics=1

### Note

This Notebook should be run in Kaggle to import the dataset.

- Competition name: dogs-vs-cats-redux-kernels-edition
- Train.zip: 569 mb
- Test.zup: 284 mb

https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition


```python

```

## Data preprocessing <a name="preprocessing"></a>

Since the Kaggle's free GPU/RAM is limited, we need to decrease the size of the raw image file so that entire traning process can be finished within the computing power.


```python
# https://www.kaggle.com/code/georgesaavedra/dogs-vs-cats-best-transfer-learning


import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import zipfile

from sklearn.model_selection import train_test_split
from tqdm import tqdm

import random

import tensorflow as tf
from sklearn.metrics import confusion_matrix
import itertools

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.optimizers import RMSprop,Adam,SGD,Adadelta
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
```


```python
# Get the list of available data
print(os.listdir("../input/dogs-vs-cats-redux-kernels-edition"))
```


```python
# Unzip the image file

with zipfile.ZipFile('../input/dogs-vs-cats-redux-kernels-edition/test.zip','r') as z:
    z.extractall('.')
    
with zipfile.ZipFile('../input/dogs-vs-cats-redux-kernels-edition/train.zip','r') as z:
    z.extractall('.')
```


```python
# Check the number of available raw image file
len(os.listdir('/kaggle/working/train/')), len(os.listdir('/kaggle/working/test/'))
```


```python
# Resize the image to reduce RAM usage and faster training

IMG_SIZE = 80
Images_train = []
Images_label = []
for i in tqdm(os.listdir('/kaggle/working/train/')):
    label = i.split('.')[0]
    if label == 'cat':
        label = 0
    elif label == 'dog':
        label = 1
    img = cv2.imread('/kaggle/working/train/'+i, cv2.IMREAD_COLOR)
    img = cv2.resize(img,(IMG_SIZE,IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    Images_train.append([np.array(img), np.array(label)])
```


```python
# Resize the image to reduce RAM usage and faster training

Images_test = []
for j in tqdm(os.listdir('/kaggle/working/test/')):
    index = j.split('.')[0]
    img = cv2.imread('/kaggle/working/test/'+j, cv2.IMREAD_COLOR)
    img = cv2.resize(img,(IMG_SIZE,IMG_SIZE), interpolation = cv2.INTER_CUBIC)
    Images_test.append([np.array(img), np.array(index)])
random.shuffle(Images_train)
```


```python
# Check the dimension of the data after the resizing

Images = np.array([i[0] for i in Images_train]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
Label = np.array([i[1] for i in Images_train])
Images.shape, Label.shape
```


```python
# Train / Validation split

X_train, X_val, Y_train, Y_val = train_test_split(Images, Label, test_size = 0.1)
X_train.shape, Y_train.shape, X_val.shape, Y_val.shape
```


```python

```

## Transfer Learning <a name="Transfer-Learning"></a>

## EfficientNetB0 Modeling

This analysis used transfer learning from 2 existing models, EfficientNetB0 and VGG 16.


```python
# Use ImageDataGenerator to fit the training data
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=10,
    zoom_range = 0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False) 
```


```python
datagen.fit(X_train)
optimizer = Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999)
```

Let's create two constraints or 'callbacks' which can help us improve the training (ReduceLROnPlateau) and stop the training once it has reached a high threshold (Callback):


```python
# Early stopping

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('val_accuracy')>0.98):
      print("\nReached 98% accuracy so cancelling training!")
      self.model.stop_training = True
        
callbacks = myCallback()

from keras.callbacks import ReduceLROnPlateau
lr_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                 patience=1, 
                                 verbose=1, 
                                 factor=0.5, 
                                 min_lr=0.000001)

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_accuracy', 
                               min_delta=0.005,
                               patience=3, 
                               verbose=1, 
                               mode='auto')
```


```python
# Complie the model

model_EF=Sequential()
model_EF.add(EfficientNetB0(input_shape=(80,80,3),
                            include_top=False,
                            pooling='max',
                            weights='imagenet'))


model_EF.layers[0].trainable=False
model_EF.add(Dense(512,activation='relu'))
model_EF.add(Dropout(0.2))
model_EF.add(Dense(1,activation='sigmoid'))
model_EF.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
model_EF.summary()
```


```python
#Train the model
history1 = model_EF.fit_generator(datagen.flow(X_train, Y_train, batch_size=32),
                                  validation_data=(X_val,Y_val), epochs=10, verbose=1,
                                  callbacks=[callbacks, lr_reduction, early_stopping])
```


```python
# Check the performance per epoch
pd.DataFrame(history1.history)
```

## VGG16 modeling


```python
# Complie the model

model_VGG=Sequential()
model_VGG.add(VGG16(input_shape=(80,80,3),
                    include_top=False,
                    pooling='max',
                    weights='imagenet'))
model_VGG.layers[0].trainable=False
model_VGG.add(Dense(512,activation='relu'))
model_VGG.add(Dropout(0.2))
model_VGG.add(Dense(1,activation='sigmoid'))
model_VGG.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
model_VGG.summary()
```


```python
# Train the model
history2 = model_VGG.fit_generator(datagen.flow(X_train, Y_train, batch_size=32),
                                  validation_data=(X_val,Y_val), epochs=20, verbose=1,
                                  callbacks=[callbacks, lr_reduction, early_stopping])

# Check the performance per epoch
pd.DataFrame(history2.history)
```


```python
# Save the model
model_VGG.save('VGG_model.h5')
```

## Prediction on testing data <a name="Prediction"></a>


Overall performace from EfficientNetB0 is better than VGG16.
So use this model to make predition


```python
X_test = np.array([j[0] for j in Images_test]).reshape(-1,IMG_SIZE, IMG_SIZE, 3)
Index = np.array([j[1] for j in Images_test])
```


```python
# Make predision

test_prediction = model_EF.predict(X_test, batch_size = 32)
```


```python
# Generate csv file for submission

submission=pd.DataFrame(test_prediction, columns=['label'], index=pd.Series(Index, name='id'))
submission.head()
```


```python
submission.to_csv('submission.csv')
```


```python

```
