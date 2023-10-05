import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import random
import zipfile
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#data unzip and load
with zipfile.ZipFile('/kaggle/input/dogs-vs-cats/train.zip', 'r') as train_zip:
    train_zip.extractall('.')  
image_dir = "../working/train/"
filenames = os.listdir(image_dir)
labels = [x.split(".")[0] for x in filenames]
data = pd.DataFrame({"filename": filenames, "label": labels})

data.head()
data['label'].value_counts()

#로드한 데이터 이미지 확인
grouped_data = data.groupby("label")

num_images_per_category = 5

fig, axes = plt.subplots(len(grouped_data), num_images_per_category, figsize = (20, 20))

for i, (category, group) in enumerate(grouped_data):
    random_indices = random.sample(range(len(group)), num_images_per_category)
    for j, index in enumerate(random_indices):
        filename = group.iloc[index]["filename"]
        label = group.iloc[index]['label']
        image_path = os.path.join(image_dir, filename)
        image = Image.open(image_path)
        
        axes[i, j].imshow(image)
        axes[i, j].set_title("Label : "+label, fontsize = 30)

plt.tight_layout()
plt.show()

#이미지 사진 값 확인
image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir)]
image_width = []
image_height = []

for image_path in image_paths:
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    image_width.append(width)
    image_height.append(height)
    
median_width = np.median(image_width)
median_height = np.median(image_height)

print("median_size : ", median_width, 'X', median_height)

#label 종류 및 갯수 확인
labels = data['label']
X_train, X_temp = train_test_split(data, test_size=0.2, stratify=labels, random_state = 23)
label_test_val = X_temp['label']
X_test, X_val = train_test_split(X_temp, test_size=0.5, stratify=label_test_val, random_state = 23)
print ('X_train:', X_train['label'].value_counts())
print ('X_val:', X_val['label'].value_counts())
print ('X_test:', X_test['label'].value_counts())

#vgg16 모델 이용
batch_size = 64
size = (370, 370)
idg = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function = tf.keras.applications.vgg16.preprocess_input)

#용도에 따른 이미지 분류
train_idg = idg.flow_from_dataframe(X_train, "train/", x_col= "filename", y_col= "label",
                                    batch_size = batch_size,
                                    target_size=size)
val_idg = idg.flow_from_dataframe(X_val, "train/", x_col="filename", y_col="label",
                                  batch_size = batch_size,
                                  target_size = size, shuffle = False)
test_idg = idg.flow_from_dataframe(X_test, "train/", x_col= "filename", y_col= "label",
                                    batch_size = batch_size,
                                    target_size=size, shuffle = False)

vgg16_model = tf.keras.applications.vgg16.VGG16(include_top=False, input_shape=(370, 370, 3))
vgg16_model.summary()

for layer in vgg16_model.layers:
  layer.trainable = False

#모델 설정
flat = tf.keras.layers.Flatten() (vgg16_model.output)
dropout1 = tf.keras.layers.Dropout(0.2, name="Dropout1") (flat)
dense1 = tf.keras.layers.Dense(128, activation="relu") (dropout1)
dropout2 = tf.keras.layers.Dropout(0.2, name="Dropout2")(dense1)
output = tf.keras.layers.Dense(2, activation="softmax") (dropout2)

final_model = tf.keras.models.Model(inputs=[vgg16_model.input], outputs=[output])

tf.keras.utils.plot_model(final_model, show_shapes = True, show_layer_names=True)

#학습 설정
final_model.compile(optimizer='adam',
                    loss=tf.keras.losses.categorical_crossentropy,
                    metrics = ["acc"])

model_ckpt = tf.keras.callbacks.ModelCheckpoint("DogCat",
                                                monitor="val_loss",
                                                save_best_only=True)

history = final_model.fit(train_idg, batch_size=batch_size, validation_data=val_idg, epochs = 8, callbacks=[model_ckpt])

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(8)

#그래프로 정확도 확인
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label = 'Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

cat_dog_model = tf.keras.models.load_model("DogCat")
result = cat_dog_model.predict(test_idg)
result_argmax = np.argmax(result, axis=1)
y_true = test_idg.labels
y_pred = result_argmax
accuracy = (y_pred == y_true).mean()
print("Test Accuracy:", accuracy)
print(classification_report(y_true, y_pred))

#test data unzip and load
with zipfile.ZipFile('/kaggle/input/dogs-vs-cats/test1.zip', 'r') as test1_zip:
    test1_zip.extractall('.')  
test_dir = "../working/test1/"
filenames = os.listdir(test_dir)
test_data = pd.DataFrame({"filename": filenames})
test_data['label'] = 'unknown'
test_data.head()

test1_idg =  idg.flow_from_dataframe(test_data, "test1/", x_col= "filename",y_col = 'label',
                                    batch_size = batch_size,
                                    target_size=size, shuffle = False)

test1_predict = cat_dog_model.predict(test1_idg)
test1_predict_argmax = np.argmax(test1_predict, axis=1)
y_test_pred = test1_predict_argmax
test_data['label'] = y_test_pred
test_data.head()
