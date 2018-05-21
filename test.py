numDimensions = 300
maxSeqLength = 200
batchSize = 24
lstmUnits = 64
numClasses = 2
iterations = 100000



import numpy as np
wordsList = np.load('wordsList.npy').tolist()
wordsList = [word.decode('UTF-8') for word in wordsList] #Encode words as UTF-8
wordVectors = np.load('wordVectors.npy')



import tensorflow as tf
tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors,input_data)

lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.25)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)

correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))


sess = tf.InteractiveSession()
saver = tf.train.Saver()


saver = tf.train.import_meta_graph('models/pretrained_lstm.ckpt-50000.meta')
saver.restore(sess,tf.train.latest_checkpoint('models'))

# Removes punctuation, parentheses, question marks, etc., and leaves only alphanumeric characters
import re
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

def getSentenceMatrix(sentence):
    arr = np.zeros([batchSize, maxSeqLength])
    sentenceMatrix = np.zeros([batchSize,maxSeqLength], dtype='int32')
    cleanedSentence = cleanSentences(sentence)
    split = cleanedSentence.split()
    for indexCounter,word in enumerate(split):
        try:
            sentenceMatrix[0,indexCounter] = wordsList.index(word)
        except ValueError:
            sentenceMatrix[0,indexCounter] = 399999 #Vector for unkown words
    return sentenceMatrix


inputText = '''Useless knowledge can be made directly contributory to a force of sound and disinterested public opinion'''



inputMatrix = getSentenceMatrix(inputText)



predictedSentiment = sess.run(prediction, {input_data: inputMatrix})[0]






final_topics = np.load('final_topics2.npy')

labels_encode={}
labels_decode={}
for i, topic in enumerate(np.unique(final_topics)):
    labels_encode[topic] = i
    labels_decode[i] = topic


index = np.where(predictedSentiment==predictedSentiment.max())[0][0]
print(labels_decode[index])






secondInputText = '''Death is the ugly fact which Nature has to hide, and she hides it well'''



secondInputMatrix = getSentenceMatrix(secondInputText)


predictedSentiment = sess.run(prediction, {input_data: secondInputMatrix})[0]



index = np.where(predictedSentiment==predictedSentiment.max())[0][0]
print(labels_decode[index])
