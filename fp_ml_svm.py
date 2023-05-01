#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np


# In[2]:


import seaborn as sns
import tensorflow as tf


# In[3]:


import os
import cv2
import random
import matplotlib.pyplot as plt


# In[4]:


from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Dropout,Activation, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.utils import to_categorical # convert to one-hot-encoding
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split


# In[5]:


def extract_label(img_path,train = True):
    filename, _ = os.path.splitext(os.path.basename(img_path))

    subject_id, etc = filename.split('__')
    
    if train:
        gender, lr, finger, _, _ = etc.split('_')
    else:
        gender, lr, finger, _ = etc.split('_')
    
    gender = 0 if gender == 'M' else 1
    lr = 0 if lr == 'Left' else 1

    if finger == 'thumb':
        finger = 0
    elif finger == 'index':
        finger = 1
    elif finger == 'middle':
        finger = 2
    elif finger == 'ring':
        finger = 3
    elif finger == 'little':
        finger = 4
    
    return np.array([subject_id, gender, lr, finger], dtype=np.uint16)
img_size = 96


# In[6]:


img_size = 96

def load_data(path,train):
    print("loading data from: ",path)
    data = []
    for img in os.listdir(path):
        try:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            img_resize = cv2.resize(img_array, (img_size, img_size))
            label = extract_label(os.path.join(path, img),train)
            data.append([label[1], img_resize ])
        except Exception as e:
            pass
    data
    return data


# In[7]:


Real_path = "C:/Users/ShivaSwaroop/Desktop/SOCOFing/SOCOFing/Real"
Easy_path = "C:/Users/ShivaSwaroop/Desktop/SOCOFing/SOCOFing/Altered/Altered-Easy"
Medium_path = "C:/Users/ShivaSwaroop/Desktop/SOCOFing/SOCOFing/Altered/Altered-Medium"
Hard_path = "C:/Users/ShivaSwaroop/Desktop/SOCOFing/SOCOFing/Altered/Altered-Hard"

easy_data = load_data(Easy_path, train = True)
medium_data = load_data(Medium_path, train = True)
hard_data = load_data(Hard_path, train = True)
test = load_data(Real_path, train = False)

data = np.concatenate([easy_data,medium_data,hard_data],axis=0)


# In[8]:


X, y = [], []

for label, feature in data:
    X.append(feature)
    y.append(label)
    
del data

X = np.array(X).reshape(-1, img_size, img_size, 1)
X = X / 255.0

y = np.array(y)


# In[9]:


X_test, y_test = [], []

for label, feature in test:
    X_test.append(feature)
    y_test.append(label)
    
del test    
X_test = np.array(X_test).reshape(-1, img_size, img_size, 1)
X_test = X_test / 255.0

y_test = np.array(y_test)


# In[10]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1)


# In[11]:


print("full data:  ",X.shape)
del X, y
print("Train:      ",X_train.shape)
print("Validation: ",X_val.shape)
print("Test:       ",X_test.shape)


# In[12]:


model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (96,96,1)))
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(100, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation = "sigmoid"))

model.summary()


# In[13]:


epochs = 5 # Turn epochs to 30 to get 0.9967 accuracy
batch_size = 32
model_path = './Model.h5'


model.compile(optimizer = 'adam' , loss = "binary_crossentropy", metrics=["accuracy"])

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=20, mode='max', verbose=1),
    ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1)
]


history = model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, 
          validation_data = (X_val, y_val), verbose = 1, callbacks= callbacks)


# In[14]:


layer_name = "dense"
feature_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

feature_model.summary()


# In[15]:


X_feature_train=feature_model.predict(X_train)


# In[16]:


X_feature_train.shape


# In[17]:


from sklearn.svm import SVC


# In[18]:


clf=SVC()
clf.fit(X_feature_train, y_train)


# In[19]:


clf.score(X_feature_train, y_train)


# In[20]:


from sklearn.metrics import confusion_matrix


# In[21]:


x_feature_val=feature_model.predict(X_val)
clf.score(x_feature_val, y_val)


# In[22]:


x_feature_test=feature_model.predict(X_test)
clf.score(x_feature_test, y_test)


# In[23]:


y_true = y_train
y_svm_pred = clf.predict(X_feature_train)
cm_svm = confusion_matrix(y_true,y_svm_pred)


# In[24]:


f,ax = plt.subplots(figsize=(8, 8))
label_map = ("male","female")
sns.heatmap(cm_svm, annot=True, linewidths=0.01,linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix on Train Set")
ax.xaxis.set_ticklabels(label_map);
ax.yaxis.set_ticklabels(label_map);
plt.show()


# In[26]:


y_true = y_test
y_svm_pred = clf.predict(x_feature_test)
cm_svm = confusion_matrix(y_true,y_svm_pred)


# In[27]:


f,ax = plt.subplots(figsize=(8, 8))
label_map = ("male","female")
sns.heatmap(cm_svm, annot=True, linewidths=0.01,linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix on Test Set")
ax.xaxis.set_ticklabels(label_map);
ax.yaxis.set_ticklabels(label_map);
plt.show()


# In[28]:


plt.figure(figsize=(24,12))

plt.suptitle("Confusion Matrixes",fontsize=24)
plt.subplots_adjust(wspace = 0.4, hspace= 0.4)


plt.subplot(2,3,3)
plt.title("Support Vector Machine Confusion Matrix")
sns.heatmap(cm_svm,annot=True,cmap="Blues",fmt="d",cbar=False, annot_kws={"size": 24})
plt.ylabel('True Label',labelpad=10)
plt.xlabel('Predicted Label',labelpad=10)
plt.show()


# In[ ]:
