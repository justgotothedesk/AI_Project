import re
import string

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import transformers
from transformers import BertTokenizer
from transformers import TFAutoModel

#data load
train = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')
test = pd.read_csv('/kaggle/input/nlp-getting-started/test.csv')

train.head()
train["length"] = train["text"].apply(lambda x : len(x))
test["length"] = test["text"].apply(lambda x : len(x))
train.head()
train["length"].describe()

train['text'] = train['text'].apply(lambda x: " ".join([word.lower() for word in str(x).split()]))
test['text'] = test['text'].apply(lambda x: " ".join([word.lower() for word in str(x).split()]))
train.head()

#html 관련 내용 제거
def remove_html(text):
    html=re.compile(r'<.*?>')
    return html.sub(r'',text)

#이모티콘 및 특수기호 내용 제거
def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

#줄임말 다시 늘리기
def decontraction(text):
    text = re.sub(r"won\'t", " will not", text)
    text = re.sub(r"won\'t've", " will not have", text)
    text = re.sub(r"can\'t", " can not", text)
    text = re.sub(r"don\'t", " do not", text)

    text = re.sub(r"can\'t've", " can not have", text)
    text = re.sub(r"ma\'am", " madam", text)
    text = re.sub(r"let\'s", " let us", text)
    text = re.sub(r"ain\'t", " am not", text)
    text = re.sub(r"shan\'t", " shall not", text)
    text = re.sub(r"sha\n't", " shall not", text)
    text = re.sub(r"o\'clock", " of the clock", text)
    text = re.sub(r"y\'all", " you all", text)

    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"n\'t've", " not have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'d've", " would have", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ll've", " will have", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    text = re.sub(r"\'re", " are", text)
    return text

#특수문자 제거
def clean(tweet):
    tweet = re.sub(r"https?:\/\/t.co\/[A-Za-z0-9]+", "", tweet)
    Special = '@#!?+&*[]-%:/()$=><|{}^'
    for s in Special:
        tweet = tweet.replace(s, "")

    return tweet

train['text'] = train['text'].apply(lambda s : clean(s))
test['text'] = test['text'].apply(lambda s : clean(s))
train['text'] = train['text'].apply(lambda s : remove_html(s))
test['text'] = test['text'].apply(lambda s : remove_html(s))
train['text'] = train['text'].apply(lambda s : remove_emoji(s))
test['text'] = test['text'].apply(lambda s : remove_emoji(s))
train['text'] = train['text'].apply(lambda s : decontraction(s))
test['text'] = test['text'].apply(lambda s : decontraction(s))

train.head()

seq_len = 256
batch_size = 16
num_samples = len(train)
model_name = 'cardiffnlp/twitter-roberta-base-sentiment'

tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

train_tokens = tokenizer(
    train['text'].tolist(),
    max_length=seq_len,
    truncation=True,
    padding='max_length',
    add_special_tokens=True,
    return_tensors='np'
)

#one hot encoding
y_train = train['target'].values
labels = np.zeros((num_samples, y_train.max() + 1))
labels[np.arange(num_samples), y_train] = 1

dataset = tf.data.Dataset.from_tensor_slices(
    (
        train_tokens['input_ids'],
        train_tokens['attention_mask'],
        labels
    )
)

def map_func(input_ids, masks, labels):
    return {
        'input_ids': input_ids,
        'attention_mask': masks
    }, labels

dataset = dataset.map(map_func)
dataset = dataset.shuffle(10000).batch(batch_size=batch_size, drop_remainder=True)

split = 0.7
size = int((train_tokens['input_ids'].shape[0] // batch_size) * split)

train_ds = dataset.take(size)
val_ds = dataset.skip(size)

#모델 및 학습 설정
model = TFAutoModel.from_pretrained(model_name)
input_ids = tf.keras.layers.Input(shape=(seq_len,), name='input_ids', dtype='int32')
mask = tf.keras.layers.Input(shape=(seq_len,), name='attention_mask', dtype='int32')

embeddings = model(input_ids, attention_mask=mask)[0]
embeddings = embeddings[:, 0, :]

x = tf.keras.layers.Dense(512, activation='relu')(embeddings)
y = tf.keras.layers.Dense(2, activation='softmax', name='outputs')(x)
bert_model = tf.keras.Model(inputs=[input_ids, mask], outputs=y)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
loss = tf.keras.losses.CategoricalCrossentropy()
acc = tf.keras.metrics.BinaryAccuracy()

bert_model.compile(optimizer=optimizer, loss=loss, metrics=[acc])

history = bert_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    batch_size=batch_size
)

bert_model.evaluate(val_ds)

def prep_data(text):
    tokens = tokenizer(
        text, max_length=256, truncation=True, 
        padding='max_length', 
        add_special_tokens=True, 
        return_tensors='tf'
    )
    return {
        'input_ids': tokens['input_ids'], 
        'attention_mask': tokens['attention_mask']
    }

test['target'] = None

for i, row in test.iterrows():
    tokens = prep_data(row['text'])
    probs = bert_model.predict_on_batch(tokens)
    pred = np.argmax(probs)
    test.at[i, 'target'] = pred
    
test['target'] = test['target'].astype(int)
test.head()

sample = pd.read_csv('/kaggle/input/nlp-getting-started/sample_submission.csv')
sub = pd.DataFrame({'id':sample['id'].values.tolist(), 'target':test['target']})
sub.head()
sub.to_csv('submission.csv', index=False)
