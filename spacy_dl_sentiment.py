import pandas as pd  # provide sql-like data manipulation tools. very handy.
from keras.callbacks import TensorBoard, EarlyStopping,ModelCheckpoint
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
from keras.layers import Embedding, LSTM, Dropout,Activation,GlobalMaxPooling1D,Conv1D,MaxPooling2D,Flatten,Convolution1D
from matplotlib import pyplot as plt

LabeledSentence = gensim.models.doc2vec.LabeledSentence  # we'll talk about this down below

from tqdm import tqdm

tqdm.pandas(desc="progress-bar")

from nltk.tokenize import TweetTokenizer  # a tweet tokenizer from nltk.

tokenizer = TweetTokenizer()

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
np.random.seed(0)
path = '/home/shaurya/datasets/AICS_Challenge_Data/AICS_Test_Data/'

#df = pd.read_csv(path + 'training_df.csv')
#labels = pd.read_csv(path + 'training_label.csv')
df = pd.read_csv(path + 'test_df.csv')
labels = pd.read_csv(path + 'test_label.csv')
# print df.describe()
# print labels.describe()

df = pd.merge(df, labels, on='uid')
labels = None

df.to_csv(path + 'title_normtext_label.csv', columns=['uid', 'title', 'normalizedText', 'label'], index=False)

print df.shape
df = None


def ingest():
    data = pd.read_csv(path + 'title_normtext_label.csv')
    data['label'] = data['label'].map(int)
    data = data[data['normalizedText'].isnull() == False]
    data = data[data['title'].isnull() == False]
    data = data[data['label'] <= 1]
    data.reset_index(inplace=True)
    data.drop('index', axis=1, inplace=True)
    print 'dataset loaded with shape', data.shape
    return data


data = ingest()
print data.head(5)
print data.describe()


tokenizer = TweetTokenizer()

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


def postprocess(data, n):
    data = data.head(n)
    data['tokens_normalizedText'] = data['normalizedText'].progress_map(
        tokenize)  ## progress_map is a variant of the map function plus a progress bar. Handy to monitor DataFrame creations.
    data = data[data.tokens_normalizedText != 'NC']
    data['tokens_title'] = data['title'].progress_map(
        tokenize)  ## progress_map is a variant of the map function plus a progress bar. Handy to monitor DataFrame creations.
    data = data[data.tokens_title != 'NC']
    data.reset_index(inplace=True)
    data.drop('index', inplace=True, axis=1)
    return data


n = data.shape[0]
n_dim = 300

data = postprocess(data, n)



x_text, y = np.array(data.head(n).tokens_normalizedText),np.array(data.head(n).label)

print x_text.shape
def labelizeTweets(tweets, label_type):
    labelized = []
    for i, v in tqdm(enumerate(tweets)):
        label = '%s_%s' % (label_type, i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized

print 'Labelize Train and Test'
x_text = labelizeTweets(x_text, 'TRAIN')
print 'building tf-idf matrix ...'
vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
matrix = vectorizer.fit_transform([x.words for x in x_text])
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
print 'vocab size :', len(tfidf)

def load_w2v():
    import numpy as np
    #path = '/home/shaurya/datasets/WikipediaClean5Negative300Skip10/WikipediaClean5Negative300Skip10.txt' - 91% accuracy
    path = '/home/shaurya/datasets/glove.6B/glove.6B.300d.txt' #91.22% 159 epoch - better loss and accuracy
    wiki_vec = {}
    for i in open(path):
        m = i.split(' ')
        word = m[0]
        vec = np.asarray(m[1:])
        vec = vec.astype(np.float)
        wiki_vec[word] = vec
    return wiki_vec

w2v_model = load_w2v()


def buildWordVector(tokens, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tokens:
        try:
            vec += w2v_model[word].reshape((1, size)) * tfidf[word]
            count += 1.
        except KeyError:  # handling the case where the token is not
            # in the corpus. useful for testing.
            continue
    if count != 0:
        vec /= count
    return vec


normtext_w2v = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, x_text))])
normtext_w2v = scale(normtext_w2v)
print normtext_w2v
#========================================================
x_text, y = np.array(data.head(n).tokens_title),np.array(data.head(n).label)
print x_text.shape

print 'Labelize Train and Test'
x_text = labelizeTweets(x_text, 'TRAIN')
print 'building tf-idf matrix ...'
vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=1)
matrix = vectorizer.fit_transform([x.words for x in x_text])
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
print 'vocab size :', len(tfidf)
title_w2v = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, x_text))])
title_w2v = scale(title_w2v)
print title_w2v
data  = np.concatenate((normtext_w2v,title_w2v),axis=1)

from scipy import spatial
similarity = []
for i,j in zip(normtext_w2v, title_w2v):
    similarity.append(1 - spatial.distance.cosine(i, j))

similarity =  np.asarray(similarity).reshape((data.shape[0],1))

data = np.concatenate((data,similarity),axis=1)

print similarity.shape
x_train, x_test, y_train, y_test = train_test_split(data,y,test_size=0.33)
print x_train.shape, y_train.shape

#Callbacks

tensorboard = TensorBoard(log_dir="/home/shaurya/PycharmProjects/fake_news/logs/{}".format(time()))
early=EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=1, mode='auto')
save_model = ModelCheckpoint('/home/shaurya/PycharmProjects/fake_news/'+'weights_1000_nodes.hdf5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)


model = Sequential()
model.add(Dense(256, activation='relu', input_dim=601))
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

hist = model.fit(x_train, y_train, epochs=10, batch_size=16, verbose=1,callbacks=[tensorboard,early,save_model], 
                 validation_data=(x_test, y_test))
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

from sklearn.metrics import roc_curve

#y_score = [val for sublist in y_score for val in sublist]
#y_score = np.asarray(y_score)

#y_score = np.rint(y_score)
#y_score = y_score.astype(int)

scores = model.evaluate(x_test, y_test, verbose=1)
print('MLP test score:', scores[0])
print('MLP test accuracy:', scores[1])
y_score = model.predict_proba(x_test)
y_score = y_score.flatten()
print y_score.flatten(), y_test
y_score = y_score.tolist()

fpr, tpr, _ = roc_curve(y_test, y_score)
from ggplot import *
df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
ggplot(df, aes(x='fpr', y='tpr')) +\
    geom_line() +\
    geom_abline(linetype='dashed')
    

auc = np.trapz(tpr,fpr)
print 'AUC: ',auc

#y_score = np.rint(y_score)
#y_score = y_score.astype(int)

#roc_x = []
#roc_y = []
#min_score = min(y_score)
#max_score = max(y_score)
#thr = np.linspace(min_score, max_score, 30)
#FP=0
#TP=0
#N = sum(y_test)
#P = len(y_test) - N
#
#for (i, T) in enumerate(thr):
#    for i in range(0, len(y_score)):
#        if (y_score[i] > T):
#            if (y_test[i]==1):
#                TP = TP + 1
#            if (y_test[i]==0):
#                FP = FP + 1
#    roc_x.append(FP/float(N))
#    roc_y.append(TP/float(P))
#    FP=0
#    TP=0
#
#plt.scatter(roc_x, roc_y)
#plt.show()
#fpr = roc_x
#tpr = roc_y
#
#print tpr
#
#
#auc = np.trapz(tpr,fpr)
#print 'AUC: ',auc






