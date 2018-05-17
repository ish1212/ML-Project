
# coding: utf-8

# In[1]:

import numpy  as  np
import tensorflow  as tf

from os import listdir
from os.path import isfile, join
import re

from random import randint
import datetime


# ## Load Data

# In[2]:

final_topics = np.load('final_topics.npy')


# In[3]:

ids = np.load('idsMatrix2.npy')


# In[4]:

ids.shape, final_topics.shape


# In[5]:

wordsList  = np.load('wordsList.npy')
wordsList = wordsList.tolist() 
wordsList = [word.decode('UTF-8') for word in wordsList] 
wordVectors = np.load('wordVectors.npy')


# ## Train-test split

# Lets make two dictionaries to go from a label to it's unique index and vice-versa:

# In[6]:

labels_encode={}
labels_decode={}
for i, topic in enumerate(np.unique(final_topics)):
    labels_encode[topic] = i
    labels_decode[i] = topic


# Now we can shuffle the ids matrix and labels together and get the training and testing lists:

# In[7]:

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(ids, final_topics, test_size=0.2 , random_state=np.random.seed(), shuffle=True)


# ## Recurrent Neural Network

# In[8]:

batchSize = 24
lstmUnits = 64
numClasses = np.unique(final_topics).size
iterations = 100000
numDimensions = 300
maxSeqLength = 200


# In[9]:

tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])


# In[10]:

data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors,input_data)


# In[11]:

lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)


# In[12]:

value  = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)

weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))

prediction = (tf.matmul(last, weight) + bias)


# In[13]:

correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))


# In[14]:

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)


# In[15]:

tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
with tf.Session() as sess:
    writer = tf.summary.FileWriter(logdir, sess.graph)


# In[16]:

def getTrainBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        num = randint(0,len(y_train)-1)
        
        one_hot = np.zeros(numClasses)
        one_hot[labels_encode[y_train[num]]] = 1
        labels.append(one_hot)
        
        arr[i] = X_train[num]
        
    return arr, labels

def getTestBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        num = randint(0,len(y_test)-1)
        
        one_hot = np.zeros(numClasses)
        one_hot[labels_encode[y_test[num]]] = 1
        labels.append(one_hot)
        
        arr[i] = X_test[num]
        
    return arr, labels


# In[17]:

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

