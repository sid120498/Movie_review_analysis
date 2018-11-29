# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 16:09:41 2018

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



#import library
import numpy as np
import pandas as pd
import os
import gc
import nltk
from tensorflow.python.keras.preprocessing import sequence, text
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Embedding, LSTM,Conv1D, GlobalMaxPooling1D,\
Flatten, MaxPooling1D, GRU, SpatialDropout1D, Bidirectional
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.losses import categorical_crossentropy
from tensorflow.python.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,\
f1_score 
import matplotlib.pyplot as plt
from tensorflow.python.keras.utils import plot_model


#importing datasets
pd.set_option('display.max_colwidth', -1)

gc.collect()

train = pd.read_csv("datasets/train.tsv", sep="\t")
test = pd.read_csv("datasets/test.tsv", sep="\t")
sub = pd.read_csv("datasets/sampleSubmission.csv")
train.shape
test.shape


#saving submission
def submission(y_pred, model):
    sub.Sentiment = y_pred
    sub.to_csv(model + '.csv', index = False)
    


#concatnating dataset
test['Sentiment'] = -999
test.head()
df = pd.concat([train, test], ignore_index = True)
del train, test
gc.collect()


#importing text preprocessing tools
#!python -m nltk.downloader all
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from nltk.stem import SnowballStemmer, WordNetLemmatizer 
stemmer = SnowballStemmer("english")
lemma = WordNetLemmatizer()
from string import punctuation
import re


#clean review
def clean_review(review_col):
    review_corpus = []
    for i in range(0, len(review_col)):
        review = str(review_col[i])
        review = re.sub('[^a-zA-Z]',' ', review)
        review = [lemma.lemmatize(w) for w in word_tokenize(str(review).lower())]
        review = ' '.join(review)
        review_corpus.append(review)
    return review_corpus

df['clean_review'] = clean_review(df.Phrase.values)
df.head(5)


#seperating train and test dataset
df_train=df[df.Sentiment!=-999]
df_train.shape

df_test=df[df.Sentiment==-999]
df_test.drop('Sentiment',axis=1,inplace=True)
df_test.shape
df_test.head()
del df
gc.collect()

#train test splitting
train_text=df_train.clean_review.values
test_text=df_test.clean_review.values
target=df_train.Sentiment.values
y=to_categorical(target)
print(train_text.shape,target.shape,y.shape)

X_train_text,X_val_text,y_train,y_val=train_test_split(train_text,y,test_size=0.2,stratify=y,random_state=123)
print(X_train_text.shape,y_train.shape)
print(X_val_text.shape,y_val.shape)

#converting model to tpu model
import os
import tensorflow as tf
def model_Tpu(model):
  TPU_WORKER = 'grpc://' + os.environ['COLAB_TPU_ADDR']
  tf.logging.set_verbosity(tf.logging.INFO)
  model = tf.contrib.tpu.keras_to_tpu_model(
  model, strategy = tf.contrib.tpu.TPUDistributionStrategy(
  tf.contrib.cluster_resolver.TPUClusterResolver(TPU_WORKER)))
  return model

#number of unique words in the train data
dist = FreqDist(word_tokenize(' '.join(X_train_text)))
NUM_UNIQUE_WORD = len(dist)

r_len=[]
for text in X_train_text:
    word=word_tokenize(text)
    l=len(word)
    r_len.append(l)
    
MAX_REVIEW_LEN=np.max(r_len)
MAX_REVIEW_LEN

#preprocessing text to
max_features = NUM_UNIQUE_WORD
max_words = MAX_REVIEW_LEN
batch_size = 64*8
epochs = 3
num_classes = 5

#training tokenizer
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train_text))
X_train = tokenizer.texts_to_sequences(X_train_text)
X_val = tokenizer.texts_to_sequences(X_val_text)
X_test = tokenizer.texts_to_sequences(test_text)


#making equal padding
X_train = sequence.pad_sequences(X_train, maxlen=max_words)
X_val = sequence.pad_sequences(X_val, maxlen=max_words)
X_test = sequence.pad_sequences(X_test, maxlen=max_words)
print(X_train.shape,X_val.shape,X_test.shape)


#defining model
model = Sequential()

model.add(Embedding(max_features, 100, input_length=max_words))
model.add(SpatialDropout1D(0.25))
model.add(Bidirectional(GRU(128)))
model.add(Dropout(0.5))

model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model = model_Tpu(model)
history= model.fit(X_train, y_train, validation_data=(X_val, y_val),epochs=100, batch_size=64*8, verbose=1)
y_pred = model.predict(X_test, verbose=1)
submission(y_pred, ",BiGru")

#predicting
y_pred4=model.predict(X_test,batch_size = 64, verbose=1)


#plotting progress
plt.plot(history.history['loss'], color='b')
plt.plot(history.history['val_loss'], color='r')
plt.show()

plt.plot(history.history['acc'], color='b')
plt.plot(history.history['val_acc'], color='r')
plt.show()


y_pred4 = np.argmax(y_pred4,axis=1)

print(X_test.shape)
for _ in range(64-X_test.shape[0]%64):
  X_test = np.concatenate((X_test, np.array([X_test[0]])))
print(X_test.shape)

y_pred4 = y_pred4[0:66292]
y_pred4.shape

submission(y_pred4, ",BiGru")