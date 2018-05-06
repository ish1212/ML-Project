
# coding: utf-8

# In[191]:

import numpy  as  np
import tensorflow  as tf
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import pandas as pd

from os import listdir
from os.path import isfile, join
import re

from random import randint
import datetime


# In[192]:

quotes_data  = pd.read_csv("quotes_all.csv",delimiter=';')


# In[193]:

quotes_data.head()


# # In[194]:
#
# quotes_data['Topic'].unique()
#
#
# # What is the distribution of Quotes over the topics?
#
# # In[195]:
#
# quotes_data['Topic'].unique().size
#
#
# # In[196]:
#
# topic_counts = []
# for topic in quotes_data['Topic'].unique():
#     topic_counts.append(sum(quotes_data['Topic']==topic))
#
#
# # In[197]:
#
# plt.hist(topic_counts)
# plt.xlabel('Frequency')
# plt.ylabel('Topic')
# plt.show()
#

# In[198]:

labels_encode={}
labels_decode={}
for i, topic in enumerate(quotes_data['Topic'].unique()):
    labels_encode[topic] = i
    labels_decode[i] = topic


# In[199]:

# print(labels_encode, labels_decode)


# In[200]:

less_topic = []
for i,count in enumerate(topic_counts):
    if count < 900:
            less_count.append(count)
            less_topic.append(labels_decode[i])
            # print(labels_decode[i], count)


len(less_topic)


# In[201]:

# counter = 0
# topic_counter = 0
# for i,count in enumerate(topic_counts):
#     if count > 900:
#             print(labels_decode[i], count)
#             counter+=count
#             topic_counter+=1
#
# print(counter, topic_counter)


# Removing some topics from DF:

# In[202]:

for topic in less_topic:
    quotes_data = quotes_data[quotes_data.Topic !=topic]


# In[232]:

quotes_data = quotes_data.reset_index(drop=True)


# In[242]:

# quotes_data.iloc[52850]


# In[309]:

# quotes_data['Topic'].unique().size


# ### Data Preprocess using Word2Vec

# In[243]:

wordsList  = np.load('wordsList.npy')
wordsList = wordsList.tolist()
wordsList = [word.decode('UTF-8') for word in wordsList]
wordVectors = np.load('wordVectors.npy')


# In[244]:

# print(len(wordsList))
# print(wordVectors.shape)
#
#
# # In[245]:
#
# print(wordsList.index('years'))
#
#
# # In[246]:
#
# numWords = []
# for quote in quotes_data['Quote']:
#     counter = 0
#     for word in quote:
#         counter+=1
#     numWords.append(counter)
#
#
# # In[247]:
#
# sum(numWords), max(numWords), min(numWords), sum(numWords)/len(numWords)
#
#
# # In[248]:
#
# quotes_data.shape[0]
#
#
# # In[249]:
#
# plt.hist(numWords, 50)
# plt.xlabel('Sequence Length')
# plt.ylabel('Frequency')
# plt.show()
#

# In[250]:

maxSeqLength = 200


# In[251]:

# strip_special_chars  = re.compile("[^A-Za-z0-9 ]+")
#
# def cleanSentences(string):
#     string = string.lower().replace("<br />", " ")
#     return re.sub(strip_special_chars, "", string.lower())


# In[252]:

# ids = np.zeros((quotes_data.shape[0], maxSeqLength), dtype='int32')

# fileCounter = 0

# id_check = []

# for quote in quotes_data['Quote']:
#     cleanedLine = cleanSentences(quote)
#     split = cleanedLine.split()
#     indexCounter = 0
#     for word in split:
#         try:
#             ids[fileCounter][indexCounter] = wordsList.index(word)
#             id_check.append(word)
#         except ValueError:
#             ids[fileCounter][indexCounter] = 399999 #Vector for unknown words
#             id_check.append(399999)
#         indexCounter = indexCounter + 1
#         if indexCounter >= maxSeqLength:
#             break
#     fileCounter = fileCounter + 1
#     print(fileCounter)
# np.save('idsMatrix2', ids)

ids = np.load('idsMatrix2.npy')


# In[253]:

# ids.shape


# ## Labels

# In[254]:

labels_encode={}
labels_decode={}
for i, topic in enumerate(quotes_data['Topic'].unique()):
    labels_encode[topic] = i
    labels_decode[i] = topic


# First we randomly shuffle our dataframe:

# In[255]:

quotes_data_shuffle = quotes_data.sample(frac=1)
quotes_data_shuffle.head()


# Store the labels:

# In[299]:

labels_y = []
for topic in quotes_data_shuffle['Topic']:
    labels_y.append(labels_encode[topic])


# In[300]:

# print(labels_y[:100])


# In[301]:
#
# len(labels_y)


# In[259]:

indices = np.asarray(quotes_data_shuffle.index)
# print(indices[0:100])


# In[335]:

# indices[0], labels_y[0]
# ids[indices[0]][:100]


# In[295]:

# quote = quotes_data_shuffle.loc[400]['Quote']
# print(quote)
#
#
# cleanedLine = cleanSentences(quote)
# split = cleanedLine.split()
# print(split)
#
# print([wordsList.index(word) for word in split])
# ids[400][:50]


# ## Recurrent Neural Network

# In[310]:

batchSize = 24
lstmUnits = 64
numClasses = quotes_data['Topic'].unique().size
iterations = 100000
numDimensions = 300


# In[311]:

tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])


# In[312]:

data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors,input_data)


# In[313]:

lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)


# In[315]:

value  = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)

weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))

prediction = (tf.matmul(last, weight) + bias)


# In[316]:

correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))


# In[317]:

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)


# In[318]:

tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
with tf.Session() as sess:
    writer = tf.summary.FileWriter(logdir, sess.graph)


# In[346]:

# labels_y[400]
# b = np.zeros(numClasses)
# b.shape
# b[labels_y[400]] = 1
# b


# In[351]:

def getTrainBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        num = randint(0,int((4*len(labels_y))/5))

        b = np.zeros(numClasses)
        b[labels_y[num]] = 1
        labels.append(b)

        arr[i] = ids[indices[num]]

    return arr, labels

def getTestBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        num = randint(int((4*len(labels_y))/5),len(labels_y))

        b = np.zeros(numClasses)
        b[labels_y[num]] = 1
        labels.append(b)

        arr[i] = ids[indices[num]]

    return arr, labels


# In[ ]:

sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

for i in range(iterations):

    nextBatch, nextBatchLabels = getTrainBatch();
    sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})

    if (i % 50 == 0):
        summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
        writer.add_summary(summary, i)

    if (i % 10000 == 0 and i != 0):
        save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
        print("saved to %s" % save_path)

writer.close()


# ## Unknown word frequency

# In[296]:

# compute how many times an unknown word has occured
#
# num_39 = 0
# id_39 = []
#
# for i, val in enumerate(ids.max(1)):
#     if val == 399999:
#         num_39+=1
#         id_39.append(i)
#
#
# # In[297]:
#
# sum(numWords), num_39


# In[ ]:
