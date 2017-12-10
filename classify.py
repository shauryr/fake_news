import pandas as pd  # provide sql-like data manipulation tools. very handy.
from keras.callbacks import TensorBoard, EarlyStopping
from time import time

pd.options.mode.chained_assignment = None
import numpy as np  # high dimensional vector computing library.
from copy import deepcopy
from string import punctuation
from random import shuffle
from sklearn.preprocessing import scale
from keras.models import Sequential
from keras.layers import Dense
import gensim
from gensim.models.word2vec import Word2Vec  # the word2vec model gensim class

LabeledSentence = gensim.models.doc2vec.LabeledSentence  # we'll talk about this down below

from tqdm import tqdm

tqdm.pandas(desc="progress-bar")

from nltk.tokenize import TweetTokenizer  # a tweet tokenizer from nltk.

tokenizer = TweetTokenizer()

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def ingest():
    data = pd.read_csv('/home/shaurya/datasets/AICS_Challenge_Data/AICS_Training_Data/output.csv')
    data['label'] = data['label'].map(int)
    data = data[data['normalizedText'].isnull() == False]
    data.reset_index(inplace=True)
    data.drop('index', axis=1, inplace=True)
    print 'dataset loaded with shape', data.shape
    return data


data = ingest()
print data.head(5)


def tokenize(tweet):
    try:
        tweet = unicode(tweet.decode('utf-8').lower())
        tokens = tokenizer.tokenize(tweet)
        tokens = filter(lambda t: not t.startswith('@'), tokens)
        tokens = filter(lambda t: not t.startswith('#'), tokens)
        tokens = filter(lambda t: not t.startswith('http'), tokens)
        return tokens
    except:
        return 'NC'


n = 20
n_dim = 300


def postprocess(data, n):
    data = data.head(n)
    data['tokens'] = data['normalizedText'].progress_map(
        tokenize)  ## progress_map is a variant of the map function plus a progress bar. Handy to monitor DataFrame creations.
    data = data[data.tokens != 'NC']
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    return data


data = postprocess(data, n)
x_train, x_test, y_train, y_test = train_test_split(np.array(data.head(n).tokens),
                                                    np.array(data.head(n).label), test_size=0.1)


def labelizeTweets(tweets, label_type):
    labelized = []
    for i, v in tqdm(enumerate(tweets)):
        label = '%s_%s' % (label_type, i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized


x_train = labelizeTweets(x_train, 'TRAIN')
x_test = labelizeTweets(x_test, 'TEST')

print 'Training Word2Vec . . .'

news_w2v = Word2Vec(size=n_dim, min_count=10)
news_w2v.build_vocab([x.words for x in tqdm(x_train)])
news_w2v.train([x.words for x in tqdm(x_train)], news_w2v.corpus_count, epochs=50)

print 'building tf-idf matrix ...'
vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
matrix = vectorizer.fit_transform([x.words for x in x_train])
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
print 'vocab size :', len(tfidf)


def buildWordVector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += news_w2v[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError:  # handling the case where the token is not
            # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec


train_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, x_train))])
train_vecs_w2v = scale(train_vecs_w2v)

test_vecs_w2v = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, x_test))])
test_vecs_w2v = scale(test_vecs_w2v)
tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=300))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

hist = model.fit(train_vecs_w2v, y_train, epochs=40, batch_size=32, verbose=1, callbacks=[tensorboard])

from matplotlib import pyplot as plt

plt.plot(hist.history['loss'], 'r')
plt.show()
