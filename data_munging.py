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

path = '/home/shaurya/datasets/AICS_Challenge_Data/AICS_Training_Data/'

df = pd.read_csv(path + 'training_df.csv')
labels = pd.read_csv(path + 'training_label.csv')
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
    data = data[data['label'] < 1]
    data.reset_index(inplace=True)
    data.drop('index', axis=1, inplace=True)
    print 'dataset loaded with shape', data.shape
    return data


data = ingest()
print data.head(5)
print data.describe()
