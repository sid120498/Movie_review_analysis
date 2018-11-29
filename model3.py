# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 16:32:38 2018

@author: Siddharth
"""

#importing datasets from kaggle
import os
"""!mkdir datasets
!pip install zipfile36
import zipfile
if(not os.path.isfile("kaggle.json")):
  from google.colab import files
  files.upload()
  !pip install -q kaggle
  !mkdir -p ~/.kaggle
  !cp kaggle.json ~/.kaggle/


"Enter datasets kaggle api"
os.chdir("datasets/")
!kaggle competitions download -c movie-review-sentiment-analysis-kernels-only
!kaggle datasets download -d yekenot/fasttext-crawl-300d-2m

#cannot extract more than 4GB
#extracting if zip exists
files = [f for f in os.listdir('.') if(os.path.isfile(f))]
for f in files:
  ext = f.split('.')[-1]
  if(ext=='zip'):
    with zipfile.ZipFile(f, 'r') as zip:
      print("unzipping "+ f)
      zip.extractall()
      os.remove(f)

os.chdir("..")"""

#importing library
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.layers import Dense, Input, LSTM, Embedding, Activation, \
Conv1D, GRU, CuDNNGRU, CuDNNLSTM, BatchNormalization, Dropout

from tensorflow.python.keras.layers import Bidirectional, GlobalMaxPool1D, Add, Flatten
from tensorflow.python.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D,\
concatenate, SpatialDropout1D
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras import initializers, regularizers, constraints, optimizers,\
layers, callbacks
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine import InputSpec, Layer
from tensorflow.python.keras.optimizers import Adam

from tensorflow.python.keras.callbacks import ModelCheckpoint, TensorBoard, Callback,\
EarlyStopping

train = pd.read_csv('datasets/train.tsv', sep="\t")
test = pd.read_csv('datasets/test.tsv', sep="\t")
sub = pd.read_csv('datasets/sampleSubmission.csv')

full_text = list(train['Phrase'].values) + list(test['Phrase'].values)

y = train['Sentiment']

tk = Tokenizer(lower = True, filters='')
tk.fit_on_texts(full_text)

train_tokenized = tk.texts_to_sequences(train['Phrase'])
test_tokenized = tk.texts_to_sequences(test['Phrase'])

max_len = 45
X_train = pad_sequences(train_tokenized, maxlen = max_len)
X_test = pad_sequences(test_tokenized, maxlen = max_len)

embedding_path = "datasets/crawl-300d-2M.vec"

embed_size = 300
max_features = 30000


def get_coefs(word,*arr): 
    return word, np.asarray(arr, dtype='float32')

embedding_index = dict(get_coefs(*o.strip().split(" ")) for o in open(embedding_path))

word_index = tk.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words + 1, embed_size))
for word, i in word_index.items():
    if i >= max_features:
        continue
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None: 
        embedding_matrix[i] = embedding_vector

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
y_ohe = ohe.fit_transform(y.values.reshape(-1, 1))

import os
import tensorflow as tf
def model_tpu(model):
  TPU_WORKER = 'grpc://' + os.environ['COLAB_TPU_ADDR']
  tf.logging.set_verbosity(tf.logging.INFO)
  tpu_model = tf.contrib.tpu.keras_to_tpu_model(
      model, strategy=tf.contrib.tpu.TPUDistributionStrategy(
			tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER)))
  return tpu_model


vocab_size = len(tk.word_index) + 1
vocab_size

def build_model(lr = 0.0, lr_d = 0.0, units = 0, spatial_dr = 0.0, kernel_size1 = 3, kernel_size2 = 2, dense_units = 12,
               dr = 0.1, conv_size = 32):
  inp = Input(shape = (max_len, ))
  x = Embedding(vocab_size, embed_size,weights = [embedding_matrix], trainable = False)(inp)
  x1 = SpatialDropout1D(spatial_dr)(x)
  x_gru = Bidirectional(GRU(units, return_sequences = True))(x1)
  x1 = Conv1D(conv_size, kernel_size = kernel_size1, padding = 'valid', kernel_initializer= 'he_uniform')(x_gru)
  avg_pool1_gru = GlobalAveragePooling1D()(x1)
  max_pool1_gru = GlobalMaxPooling1D()(x1)
  x3 = Conv1D(conv_size, kernel_size = kernel_size2, padding = 'valid', kernel_initializer = 'he_uniform')(x_gru)
  avg_pool3_gru = GlobalAveragePooling1D()(x3)
  max_pool3_gru = GlobalMaxPooling1D()(x3)
  
  x_lstm = Bidirectional(LSTM(units, return_sequences = True))(x1)
  x1 = Conv1D(conv_size, kernel_size=kernel_size1, padding='valid', kernel_initializer='he_uniform')(x_lstm)
  avg_pool1_lstm = GlobalAveragePooling1D()(x1)
  max_pool1_lstm = GlobalMaxPooling1D()(x1)
  x3 = Conv1D(conv_size, kernel_size=kernel_size2, padding='valid', kernel_initializer='he_uniform')(x_lstm)
  avg_pool3_lstm = GlobalAveragePooling1D()(x3)
  max_pool3_lstm = GlobalMaxPooling1D()(x3)
  
  x = concatenate([avg_pool1_gru, max_pool1_gru, avg_pool3_gru, max_pool3_gru,
                    avg_pool1_lstm, max_pool1_lstm, avg_pool3_lstm, max_pool3_lstm])
  
  x = BatchNormalization()(x)
  x = Dense(dense_units//2, activation = 'relu')(x)
  x = Dropout(dr)(x)
  x = Dense(5, activation='sigmoid')(x)
  model = Model(inputs = inp, outputs = x)
  model.compile(loss = "categorical_crossentropy", optimizer = Adam(lr = lr, decay = lr_d, clipvalue = 5),
                metrics = ["accuracy"])
  model = model_tpu(model)
  
  history = model.fit(X_train, y_ohe, batch_size = 128 * 8, epochs = 1000, validation_split=0.1, 
                        verbose = 1)
  return model, history

model,history = build_model(lr = 1e-3, lr_d = 1e-10,
                      units = 64, spatial_dr = 0.3, kernel_size1=3,
                      kernel_size2=2, dense_units=32, dr=0.1, conv_size=32)

import matplotlib.pyplot as plt
plt.plot(history.history['loss'], color='b')
plt.plot(history.history['val_loss'], color='r')
plt.show()

plt.plot(history.history['acc'], color='b')
plt.plot(history.history['val_acc'], color='r')
plt.show()

for _ in range(64-X_test.shape[0]%64):
  X_test = np.concatenate((X_test, np.array([X_test[0]])))
print(X_test.shape)


y_pred = model.predict(X_test, batch_size  = 64, verbose = 1)

y_pre = np.argmax(y_pred, axis = 1)

sub['Sentiment'] = y_pre[0:66292]
sub.to_csv("ans" + '.csv', index = False)