import numpy as np
import pandas as pd

import tensorflow as tf

# Parameters
max_titleSeqLength = 128
max_textSeqLength = 256
Unknown_indx = 399999

batchSize = 32
lstmUnits = 64
numClasses = 2
iterations = 100000

path = '/home/shaurya/datasets/glove.6B/glove.6B.300d.txt'
wordsList = []
wordVectors = []
for i in open(path):
    m = i.split(' ')
    word = m[0]
    vec = np.asarray(m[1:])
    vec = vec.astype(np.float)
    wordsList.append(word)
    wordVectors.append(vec)
    if len(wordsList) % 10000 == 0:
        print '%d words done!' % (len(wordsList))

print 'All words done'
wordsList = np.array(wordsList, dtype=str)
wordsList = [word.decode('UTF-8') for word in wordsList]  # Encode words as UTF-8
wordVectors = np.array(wordVectors, dtype=np.float32)
print 'Converted to NumPy arrays'


def get_word_indexes(data, max_SeqLength, Unkown_indx):
    num_samples = data.shape[0]
    idxs = np.zeros((num_samples, max_SeqLength), dtype='int32')
    for sample in range(num_samples):
        sentence = data[sample]
        # Extracting title indexes
        for i in range(len(sentence)):
            word = sentence[i]
            if i >= max_SeqLength: break
            try:
                idxs[sample, i] = wordsList.index(word)
            except:
                idxs[sample, i] = Unknown_indx
    return idxs


print 'Word Vectors Created!'

# Creating Train and Test indexes
# df = pd.read_csv('../../../training_df.csv')
# train_labels = pd.read_csv('../../../training_label.csv')['label']
df = pd.read_csv('/home/shaurya/datasets/AICS_Challenge_Data/AICS_Training_Data/training_df.csv')
train_labels = pd.read_csv('/home/shaurya/datasets/AICS_Challenge_Data/AICS_Training_Data/training_label.csv')['label']
train_title_idxs = get_word_indexes(df['title'], max_titleSeqLength, Unknown_indx)
train_text_idxs = get_word_indexes(df['normalizedText'], max_textSeqLength, Unknown_indx)

# df = pd.read_csv('../../../testing_df.csv')
# test_labels = pd.read_csv('../../../testing_label.csv')['label']
df = pd.read_csv('/home/shaurya/datasets/AICS_Challenge_Data/AICS_Testing_Data/testing_df.csv')
test_labels = pd.read_csv('/home/shaurya/datasets/AICS_Challenge_Data/AICS_Testing_Data/testing_label.csv')['label']
test_title_idxs = get_word_indexes(df['title'], max_titleSeqLength, Unknown_indx)
test_text_idxs = get_word_indexes(df['normalizedText'], max_textSeqLength, Unknown_indx)

# Utilities
from random import randint


def get_nextBatch(batchSize, title_idxs,
                  text_idxs, labels,
                  max_titleSeqLength, max_textSeqLength):
    num_samples = title_idxs.shape[0]
    batch_titles = np.zeros([batchSize, max_titleSeqLength])
    batch_texts = np.zeros([batchSize, max_textSeqLength])
    batch_labels = np.zeros([batchSize, ])
    for i in range(batchSize):
        num = randint(range(int(num_samples)))
        batch_titles[i] = title_idxs[num]
        batch_texts[i] = text_idxs[num]
        batch_labels = labels[num]
    return batch_titles, batch_texts, batch_labels


########################### TensorFlow computational graph #######################
tf.reset_default_graph()

true_label = tf.placeholder(tf.float32, [batchSize, numClasses])
title_input = tf.placeholder(tf.int32, [batchSize, max_titleSeqLength])
text_input = tf.placeholder(tf.int32, [batchSize, max_textSeqLength])

# Creating the word vectors
title_word_vecs = tf.Variable(tf.nn.embedding_lookup(wordVectors, title_input), dtype=tf.float32)
text_word_vecs = tf.Variable(tf.nn.embedding_lookup(wordVectors, text_input), dtype=tf.float32)

# Building the LSTM Cell
title_lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
title_lstmCell = tf.contrib.rnn.DropoutWrapper(cell=title_lstmCell, output_keep_prob=0.75)
text_lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
text_lstmCell = tf.contrib.rnn.DropoutWrapper(cell=text_lstmCell, output_keep_prob=0.75)

with tf.variable_scope('title_lstm'):
    title_seq, _ = tf.nn.dynamic_rnn(title_lstmCell, title_word_vecs,
                                     dtype=tf.float32)  # Shape: BatchSize x max_titleSeqLength x lstmUnits
with tf.variable_scope('text_lstm'):
    text_seq, _ = tf.nn.dynamic_rnn(text_lstmCell, text_word_vecs,
                                    dtype=tf.float32)  # Shape: BatchSize x max_textSeqLength x lstmUnits
# text_seq = title_seq
title_seq = tf.transpose(title_seq, [1, 0, 2])  # Shape: max_title_SeqLength x BatchSize x lstmUnits
text_seq = tf.transpose(text_seq, [1, 0, 2])  # Shape: max_text_Seq_Length x BatchSize x lstmUnits

title_enc = tf.gather(title_seq, int(title_seq.get_shape()[0]) - 1)  # Shape: BatchSize x lstmUnits
text_enc = tf.gather(text_seq, int(text_seq.get_shape()[0]) - 1)  # Shape: BatchSize x lstmUnits

text_title_concat = tf.concat([title_enc, text_enc], axis=1)  # Shape: BatchSize x (2*lstmUnits)

weight = tf.Variable(tf.truncated_normal([2 * lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))

pred_label = tf.matmul(text_title_concat, weight) + bias

correctPred = tf.equal(tf.argmax(pred_label, 1), tf.argmax(true_label, 1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_label, labels=true_label))
optimizer = tf.train.AdamOptimizer().minimize(loss)

# TensorBoard
import datetime

with tf.Session() as sess:
    tf.summary.scalar('Loss', loss)
    tf.summary.scalar('Accuracy', accuracy)
    merged = tf.summary.merge_all()
    logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    writer = tf.summary.FileWriter(logdir, sess.graph)

########################## Training #########################
sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
print 'RUNNING Tensorflow'
for i in range(iterations):
    # Next Batch
    nextBatch_titles, nextBatch_texts, nextBatch_labels = get_nextBatch(batchSize, train_title_idxs,
                                                                        train_text_idxs, train_labels,
                                                                        max_titleSeqLength, max_textSeqLength);
    sess.run(optimizer, {title_input: nextBatch_titles,
                         text_input: nextBatch_texts,
                         true_label: nextBatch_labels})

    # Write summary to Tensorboard
    if (i % 50 == 0):
        summary = sess.run(merged, {title_input: nextBatch_titles,
                                    text_input: nextBatch_texts,
                                    true_label: nextBatch_labels})
        writer.add_summary(summary, i)

    # Save the network every 10,000 training iterations
    if (i % 10000 == 0 and i != 0):
        save_path = saver.save(sess, "models/FakeNews_lstm.ckpt", global_step=i)
        print("saved to %s" % save_path)
writer.close()

print 'AUC'
# Code for AUC curve
from sklearn.metrics import roc_curve

iterations = 10
for i in range(iterations):
    nextBatch_titles, nextBatch_texts, nextBatch_labels = get_nextBatch(batchSize, test_title_idxs,
                                                                        test_text_idxs, test_labels,
                                                                        max_titleSeqLength, max_textSeqLength);
    pred_test_labels, test_accuracy = sess.run(pred_label, accuracy, {title_input: nextBatch_titles,
                                                                      text_input: nextBatch_texts,
                                                                      true_label: nextBatch_labels})
    fpr, tpr, _ = roc_curve(nextBatch_labels, pred_test_labels, pos_label=2)

    auc = np.trapz(tpr, fpr)
    print 'AUC: ', auc

# plt.plot(fpr, tpr)
# plt.show()
