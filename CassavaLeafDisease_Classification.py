import os
import glob
import shutil
import json
import keras
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
from keras.preprocessing.image import ImageDataGenerator

#data load
work_dir = '../input/cassava-leaf-disease-classification/'
os.listdir(work_dir) 
train_data = '/kaggle/input/cassava-leaf-disease-classification/train_images'
data = pd.read_csv(work_dir + 'train.csv')
#label 빈도 출력
print(Counter(data['label']))

#load json file
f = open(work_dir + 'label_num_to_disease_map.json')
real_labels = json.load(f)
real_labels = {int(k):v for k,v in real_labels.items()}

#label mapping
data['class_name'] = data.label.map(real_labels)

#train, val data 나누기
from sklearn.model_selection import train_test_split
train,val = train_test_split(data, test_size = 0.05, random_state = 42, stratify = data['class_name'])

IMG_SIZE = 456
size = (IMG_SIZE,IMG_SIZE)
n_CLASS = 5
BATCH_SIZE = 15

#train data만 argumentation
datagen_train = ImageDataGenerator(
                    preprocessing_function = tf.keras.applications.efficientnet.preprocess_input,
                    rotation_range = 40,
                    width_shift_range = 0.2,
                    height_shift_range = 0.2,
                    shear_range = 0.2,
                    zoom_range = 0.2,
                    horizontal_flip = True,
                    vertical_flip = True,
                    fill_mode = 'nearest')

datagen_val = ImageDataGenerator(
                    preprocessing_function = tf.keras.applications.efficientnet.preprocess_input,
                    )

train_set = datagen_train.flow_from_dataframe(train,
                             directory = train_path,
                             seed=42,
                             x_col = 'image_id',
                             y_col = 'class_name',
                             target_size = size,
                             class_mode = 'categorical',
                             interpolation = 'nearest',
                             shuffle = True,
                             batch_size = BATCH_SIZE)

val_set = datagen_val.flow_from_dataframe(val,
                             directory = train_path,
                             seed=42,
                             x_col = 'image_id',
                             y_col = 'class_name',
                             target_size = size,
                             class_mode = 'categorical',
                             interpolation = 'nearest',
                             shuffle = True,
                             batch_size = BATCH_SIZE)

#모델 설정
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.optimizers import RMSprop, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB3

def create_model():
    model = Sequential()
    model.add(EfficientNetB3(input_shape = (IMG_SIZE, IMG_SIZE, 3), include_top = False,
                             weights = 'imagenet',
                             drop_connect_rate=0.6))
    model.add(GlobalAveragePooling2D())
    model.add(Flatten())
    model.add(Dense(256, activation = 'relu', bias_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(n_CLASS, activation = 'softmax'))
    
    return model

leaf_model = create_model()
leaf_model.summary()

keras.utils.plot_model(leaf_model)

#헉습 조건 설정
EPOCHS = 50
STEP_SIZE_TRAIN = train_set.n//train_set.batch_size
STEP_SIZE_VALID = val_set.n//val_set.batch_size

def Model_fit():    
    leaf_model = create_model()    
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits = False,
                                                   label_smoothing = 0.0001,
                                                   name = 'categorical_crossentropy' )
    leaf_model.compile(optimizer = Adam(learning_rate = 1e-3),
                        loss = loss,
                        metrics = ['categorical_accuracy'])
  
    #val_loss가 3 epochs 동안 줄지 않으면 early stop
    es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 3,
                       restore_best_weights = True, verbose = 1)
    
    #minimum validation loss 해당 모델 저장
    checkpoint_cb = ModelCheckpoint("Cassava_best_model.h5",
                                    save_best_only = True,
                                    monitor = 'val_loss',
                                    mode = 'min')
    
    #학습률 조정
    reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                                  factor = 0.2,
                                  patience = 2,
                                  min_lr = 1e-6,
                                  mode = 'min',
                                  verbose = 1)
    
    history = leaf_model.fit(train_set,
                             validation_data = val_set,
                             epochs = EPOCHS,
                             batch_size = BATCH_SIZE,
                             steps_per_epoch = STEP_SIZE_TRAIN,
                             validation_steps = STEP_SIZE_VALID,
                             callbacks = [es, checkpoint_cb, reduce_lr])

    #모델 저장
    leaf_model.save('Cassava_model'+'.h5')  
    
    return history

#학습 시작
results = Model_fit()

#train data는 약 90%, valid data는 약 89.2%의 정확도가 나왔음
print('train data accuracy: ', max(results.history['categorical_accuracy']))
print('valid data accuracy: ', max(results.history['val_categorical_accuracy']))

#그래프를 통한 정확도 및 loss 값 시각화
def Train_Val_Plot(acc,val_acc,loss,val_loss):    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize= (15,10))
    fig.suptitle("정확도", fontsize=20)

    ax1.plot(range(1, len(acc) + 1), acc)
    ax1.plot(range(1, len(val_acc) + 1), val_acc)
    ax1.set_title('History of Accuracy', fontsize=15)
    ax1.set_xlabel('Epochs', fontsize=15)
    ax1.set_ylabel('Accuracy', fontsize=15)
    ax1.legend(['training', 'validation'])

    ax2.plot(range(1, len(loss) + 1), loss)
    ax2.plot(range(1, len(val_loss) + 1), val_loss)
    ax2.set_title('History of Loss', fontsize=15)
    ax2.set_xlabel('Epochs', fontsize=15)
    ax2.set_ylabel('Loss', fontsize=15)
    ax2.legend(['training', 'validation'])
    plt.show()

#train data는 딱히 문제가 없는 형태였으나, valid data는 loss 값이 3에서 급격하게 상승했다가 4에서 다시 급격하게 감소함. 아마 train data의 argumentation을 수정해야할 듯함.
Train_Val_Plot(results.history['categorical_accuracy'],results.history['val_categorical_accuracy'],
               results.history['loss'],results.history['val_loss'])

import keras
#model load
final_model = keras.models.load_model('Cassava_best_model.h5')

TEST_DIR = '../input/cassava-leaf-disease-classification/test_images/'
test_images = os.listdir(TEST_DIR)
datagen = ImageDataGenerator(horizontal_flip=True)

#model test
def pred(images):
    for image in test_images:
        img = Image.open(TEST_DIR + image)
        img = img.resize(size)
        samples = np.expand_dims(img, axis=0)
        it = datagen.flow(samples, batch_size=10)
        yhats = final_model.predict_generator(it, steps=10, verbose=0)
        summed = np.sum(yhats, axis=0)
    return np.argmax(summed)

predictions = pred(test_images)

sub = pd.DataFrame({'image_id': test_images, 'label': predictions})
display(sub)
sub.to_csv('submission.csv', index = False)
