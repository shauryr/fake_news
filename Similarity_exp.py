import pandas as pd  # provide sql-like data manipulation tools. very handy.
from time import time
from nltk.tokenize import TweetTokenizer  # a tweet tokenizer from nltk.
from tqdm import tqdm
import numpy as np  # high dimensional vector computing library.
np.random.seed(0)
from copy import deepcopy
from keras.callbacks import TensorBoard, EarlyStopping,ModelCheckpoint
from string import punctuation
from random import shuffle
from sklearn.preprocessing import scale
from keras.models import Sequential
from keras.layers import Dense
import gensim
from gensim.models.word2vec import Word2Vec  # the word2vec model gensim class
from keras.layers import Embedding, LSTM, Dropout,Activation,GlobalMaxPooling1D,Conv1D,MaxPooling2D,Flatten,Convolution1D
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

pd.options.mode.chained_assignment = None

LabeledSentence = gensim.models.doc2vec.LabeledSentence  # we'll talk about this down below

tqdm.pandas(desc="progress-bar")

tokenizer = TweetTokenizer()


path = '/home/shaurya/datasets/AICS_Challenge_Data/'

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


def ingest(path_to_data):
    data = pd.read_csv(path_to_data + 'title_normtext_label.csv')
    data['label'] = data['label'].map(int)
    data = data[data['normalizedText'].isnull() == False]
    data = data[data['title'].isnull() == False]
    data = data[data['label'] <= 1]
    data.reset_index(inplace=True)
    data.drop('index', axis=1, inplace=True)
    print 'dataset loaded with shape', data.shape
    return data


data_train = ingest(path+'AICS_Training_Data/')
data_test = ingest(path+'AICS_Test_Data/')


tokenizer = TweetTokenizer()

def tokenize(tweet):
    try:
        tweet = unicode(tweet.decode('utf-8').lower())
        tokens = tokenizer.tokenize(tweet)
        #tokens = filter(lambda t: not t.startswith('@'), tokens)
        #tokens = filter(lambda t: not t.startswith('#'), tokens)
        #tokens = filter(lambda t: not t.startswith('http'), tokens)
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


n_train = data_train.shape[0]
n_test = data_test.shape[0]
n_dim = 300

data_train = postprocess(data_train, n_train)
data_test = postprocess(data_test, n_test)


x_text_train, y_train = np.array(data_train.head(n_train).tokens_normalizedText),np.array(data_train.head(n_train).label)
x_text_test, y_test = np.array(data_test.head(n_test).tokens_normalizedText),np.array(data_test.head(n_test).label)

def labelizeTweets(tweets, label_type):
    labelized = []
    for i, v in tqdm(enumerate(tweets)):
        label = '%s_%s' % (label_type, i)
        labelized.append(LabeledSentence(v, [label]))
    return labelized

print 'Labelize Train and Test'
x_text_train = labelizeTweets(x_text_train, 'TRAIN')
x_text_test = labelizeTweets(x_text_test, 'TEST')

print 'building tf-idf matrix ... TRAIN'
vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
matrix = vectorizer.fit_transform([x.words for x in x_text_train])
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
print 'vocab size :', len(tfidf)


print 'building tf-idf matrix ... TEST'
vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=10)
matrix = vectorizer.fit_transform([x.words for x in x_text_test])
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
print 'vocab size :', len(tfidf)


def load_w2v():
    import numpy as np
    #path = '/home/shaurya/datasets/WikipediaClean5Negative300Skip10/WikipediaClean5Negative300Skip10.txt' # - 91% accuracy - 92.1553876919 AUC 10 poch
    #path = '/home/shaurya/datasets/glove.6B/glove.6B.300d.txt' #91.22% 159 epoch - better loss and accuracy
    path = '/home/shaurya/datasets/wiki.en/wiki.en.txt'
    wiki_vec = {}
    for i in tqdm(open(path)):
        m = i.strip().split(' ')
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


normtext_w2v_train = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, x_text_train))])
normtext_w2v_train = scale(normtext_w2v_train)


normtext_w2v_test = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, x_text_test))])
normtext_w2v_test = scale(normtext_w2v_test)



#print normtext_w2v.shape
#========================================================
#x_text, y = np.array(data.head(n).tokens_title),np.array(data.head(n).label)

x_text_train, y_train = np.array(data_train.head(n_train).tokens_title),np.array(data_train.head(n_train).label)
x_text_test, y_test = np.array(data_test.head(n_test).tokens_title),np.array(data_test.head(n_test).label)




print 'Labelize Train and Test'
x_text_train = labelizeTweets(x_text_train, 'TRAIN')
x_text_test = labelizeTweets(x_text_test, 'TEST')

print 'building tf-idf matrix ...'
vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=1)
matrix = vectorizer.fit_transform([x.words for x in x_text_train])
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))





print 'vocab size :', len(tfidf)
title_w2v_train = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, x_text_train))])
title_w2v_train = scale(title_w2v_train)

title_w2v_test = np.concatenate([buildWordVector(z, n_dim) for z in tqdm(map(lambda x: x.words, x_text_test))])
title_w2v_test = scale(title_w2v_test)

w2v_model = None
data_2  = np.concatenate((normtext_w2v_train,title_w2v_train),axis=1)
#
#from scipy import spatial
#similarity = []
#for i,j in zip(normtext_w2v, title_w2v):
#    similarity.append(1 - spatial.distance.cosine(i, j))
#
#similarity =  np.asarray(similarity).reshape((data_2.shape[0],1))
#
#data_2 = np.concatenate((data_2,similarity),axis=1)
#
#print similarity.shape
x_train, x_test, y_train, y_test = train_test_split(data_2,y,test_size=0.33)
print x_train.shape, y_train.shape

#Callbacks

tensorboard = TensorBoard(log_dir="/home/shaurya/PycharmProjects/fake_news/logs/{}".format(time()))
early=EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=1, mode='auto')
save_model = ModelCheckpoint('/home/shaurya/PycharmProjects/fake_news/'+'weights_removingcosine.hdf5', monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)


model = Sequential()
model.add(Dense(256, activation='relu', input_dim=600))
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

hist = model.fit(x_train, y_train, epochs=25, batch_size=16, verbose=1,callbacks=[tensorboard,early,save_model],
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

from sklearn.metrics import roc_curve, auc,precision_recall_curve

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

plt.plot(fpr,tpr)
plt.show

from ggplot import *
df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
print ggplot(df, aes(x='fpr', y='tpr')) +\
    geom_line() +\
    geom_abline(linetype='dashed')

auc_score = np.trapz(tpr,fpr)
print 'AUC: ',auc_score

roc_auc = auc(fpr, tpr)
#
#fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
#roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='MLP (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()



from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_test, y_score)

print('Average precision-recall score: {0:0.2f}'.format(
      average_precision))

precision, recall, _ = precision_recall_curve(y_test, y_score)

plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2,
                 color='b')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(
          average_precision))



from sklearn.metrics import recall_score
recall_score(y_test, np.array(y_score).astype(int), average='macro')

recall_score(y_test, np.array(y_score).astype(int), average='micro')
