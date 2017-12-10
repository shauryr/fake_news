import pandas as pd  # provide sql-like data manipulation tools. very handy.
from keras.callbacks import TensorBoard, EarlyStopping
from time import time
import gensim
pd.options.mode.chained_assignment = None
import numpy as np  # high dimensional vector computing library.
from copy import deepcopy
from string import punctuation
from random import shuffle
from sklearn.preprocessing import scale
from keras.models import Sequential
from keras.layers import Dense,Embedding, LSTM, Dropout,Activation,GlobalMaxPooling1D,Conv1D,MaxPooling2D,Flatten,Convolution1D
import gensim
from gensim.models.word2vec import Word2Vec  # the word2vec model gensim class
import keras
LabeledSentence = gensim.models.doc2vec.LabeledSentence  # we'll talk about this down below

from tqdm import tqdm

tqdm.pandas(desc="progress-bar")

from nltk.tokenize import TweetTokenizer  # a tweet tokenizer from nltk.

tokenizer = TweetTokenizer()

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

n_train = 130000
n_dim = 300
n_test = 140000
print 'Reading Train and Test Data . .'
data_train=pd.read_pickle('/home/shaurya/datasets/AICS_Challenge_Data/data_train', compression='infer')
data_test=pd.read_pickle('/home/shaurya/datasets/AICS_Challenge_Data/data_test', compression='infer')

x_train, y_train = np.array(data_train.head(n_train).tokens), np.array(data_train.head(n_train).label)
x_test, y_test = np.array(data_test.head(n_test).tokens), np.array(data_test.head(n_test).label)

def labelizeTweets(tweets, label_type):
    labelized = []
    for i, v in tqdm(enumerate(tweets)):
        label = '%s_%s' % (label_type, i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized

print 'Labelize Train and Test'
x_train = labelizeTweets(x_train, 'TRAIN')
x_test = labelizeTweets(x_test, 'TEST')

#print 'Training Word2Vec . . .'

#news_w2v = Word2Vec(size=n_dim, min_count=10)
#news_w2v.build_vocab([x.words for x in tqdm(x_train)])
#news_w2v.train([x.words for x in tqdm(x_train)], news_w2v.corpus_count, epochs=50)

print 'building tf-idf matrix ...'
vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
matrix = vectorizer.fit_transform([x.words for x in x_train])
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
print 'vocab size :', len(tfidf)

#news_w2v = Word2Vec.load_word2vec_format('/home/shaurya/datasets/AICS_Challenge_Data/Word_emb',compression='infer')
#news_w2v = gensim.models.KeyedVectors.load_word2vec_format('/home/shaurya/datasets/AICS_Challenge_Data/Word_emb',binary=True, unicode_errors='ignore')

import spacy
nlp = spacy.load('en')

def buildWordVector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += nlp(word).vector.reshape((1,size))* tfidf[word]
            #vec += news_w2v[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError or UnicodeEncodeError:  # handling the case where the token is not
            # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec
print 'Building train and test vecs from spacy'
train_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, x_train))])
train_vecs_w2v = scale(train_vecs_w2v)

test_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, x_test))])
test_vecs_w2v = scale(test_vecs_w2v)

tensorboard = TensorBoard(log_dir="./logs/{}".format(time()))

model = Sequential()
model.add(Dense(256, activation='relu', input_dim=n_dim))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

hist = model.fit(train_vecs_w2v, y_train, epochs=60, batch_size=32, verbose=1,callbacks=[tensorboard], 
                 validation_data=(test_vecs_w2v, y_test))
#
#top_words = 10000
#max_review_length = 1600
#embedding_vecor_length = 300
#model = Sequential()
#model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
#model.add(Conv1D(64, 3, border_mode='same'))
#model.add(Conv1D(32, 3, border_mode='same'))
#model.add(Conv1D(16, 3, border_mode='same'))
#model.add(Flatten())
#model.add(Dropout(0.2))
#model.add(Dense(180,activation='sigmoid'))
#model.add(Dropout(0.2))
#model.add(Dense(1,activation='sigmoid'))
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
#
#hist = model.fit(x_train, y_train, epochs=60, batch_size=32, verbose=1,callbacks=[tensorboard], 
#                 validation_data=(x_test, y_test))
from matplotlib import pyplot as plt

plt.plot(hist.history['loss'], 'r')
plt.show()

scores = model.evaluate(test_vecs_w2v, y_test, verbose=1)
print('MLP test score:', scores[0])
print('MLP test accuracy:', scores[1])

