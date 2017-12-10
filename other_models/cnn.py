'''This script loads pre-trained word embeddings (GloVe embeddings)
into a frozen Keras Embedding layer, and uses it to
train a text classification model on the 20 Newsgroup dataset
(classification of newsgroup messages into 20 different categories).

GloVe embedding data can be found at:
http://nlp.stanford.edu/data/glove.6B.zip
(source page: http://nlp.stanford.edu/projects/glove/)

20 Newsgroup data can be found at:
http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html
'''

from __future__ import print_function
from keras.models import Sequential

import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding, SimpleRNN, LSTM, Bidirectional, Dropout
from keras.models import Model
import pandas as pd
from tqdm import tqdm

BASE_DIR = '/home/shaurya/datasets/'
GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
# TEXT_DATA_DIR = os.path.join(BASE_DIR, '20_newsgroup')
MAX_SEQUENCE_LENGTH_TITLE = 1000
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 100
# VALIDATION_SPLIT = 0.2
DATA_PATH_TRAIN = os.path.join(BASE_DIR, 'AICS_Challenge_Data/AICS_Training_Data/')
DATA_PATH_TEST = os.path.join(BASE_DIR, 'AICS_Challenge_Data/AICS_Test_Data/')
# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in tqdm(f):
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# second, prepare text samples and their labels
print('Processing text dataset')

texts_train = []  # list of text samples
labels_train = []  # list of label ids

texts_test = []  # list of text samples
labels_test = []  # list of label ids

df_train = pd.read_csv(os.path.join(DATA_PATH_TRAIN, 'training_df.csv'), dtype={'title': object})
df_train_label = pd.read_csv(os.path.join(DATA_PATH_TRAIN, 'training_label.csv'))

df_test = pd.read_csv(os.path.join(DATA_PATH_TEST, 'test_df.csv'), dtype={'title': object})
df_test_label = pd.read_csv(os.path.join(DATA_PATH_TEST, 'test_label.csv'))

for normalizedText, label in tqdm(zip(df_train['normalizedText'], df_train_label['label'])):
    if label == 0 or label == 1:
        texts_train.append(str(normalizedText))
        labels_train.append(label)

for normalizedText, label in tqdm(zip(df_test['normalizedText'], df_test_label['label'])):
    if label == 0 or label == 1:
        texts_test.append(str(normalizedText))
        labels_test.append(label)

labels_index = list(set(labels_train))
print('Found %s texts.' % len(texts_train))

# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts_train)
sequences_train = tokenizer.texts_to_sequences(texts_train)
sequences_test = tokenizer.texts_to_sequences(texts_test)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data_train = pad_sequences(sequences_train, maxlen=MAX_SEQUENCE_LENGTH_TITLE)
data_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH_TITLE)

labels_train = to_categorical(np.asarray(labels_train))
labels_test = to_categorical(np.asarray(labels_test))

print('Shape of data tensor:', data_train.shape)
print('Shape of label tensor:', labels_train.shape)

x_train = data_train
y_train = labels_train
x_val = data_test
y_val = labels_test

print('Preparing embedding matrix.')

# prepare embedding matrix
num_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in tqdm(word_index.items()):
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector  # load pre-trained word embeddings into an Embedding layer
        # note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH_TITLE,
                            trainable=False)

print('Training model.')


def model_cnn():
    # train a 1D convnet with global maxpooling
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH_TITLE,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(len(labels_index), activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    model.fit(x_train, y_train,
              batch_size=128,
              epochs=10,
              validation_data=(x_val, y_val))


def model_rnn():
    print('Build model...')
    model = Sequential()
    model.add(Embedding(MAX_SEQUENCE_LENGTH_TITLE, 128))
    model.add(SimpleRNN(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2, activation='sigmoid'))
    # try using different optimizers and different optimizer configs
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    print('Train...')
    model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_val, y_val))


def mode_birnn():
    model = Sequential()
    model.add(Embedding(MAX_SEQUENCE_LENGTH_TITLE, 128, input_length=MAX_SEQUENCE_LENGTH_TITLE))
    model.add(Bidirectional(SimpleRNN(64)))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='sigmoid'))
    # try using different optimizers and different optimizer configs
    model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
    model.summary()
    print('Train...')
    model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_val, y_val))


# if __name__ == '__main__':
model_cnn()
# model_rnn()
# mode_birnn()
