import sys
import os
import numpy as np
import tensorflow as tf


with open('trump.txt', 'r', encoding='utf-8') as fp:
    txt = fp.read()

tf.reset_default_graph()

vocab = list(set(txt))
len(txt), len(vocab)

encoder = dict(zip(vocab, range(len(vocab))))
decoder = dict(zip(range(len(vocab)), vocab))

# Number of sequences in a mini batch
batch_size = 100

# Number of characters in a sequence
sequence_length = 100

# Number of cells in our LSTM layer
n_cells = 256

# Number of LSTM layers
n_layers = 2

# Total number of characters in the one-hot encoding
n_chars = len(vocab)


X = tf.placeholder(tf.int32, [None, sequence_length], name='X')

# We'll have a placeholder for our true outputs
Y = tf.placeholder(tf.int32, [None, sequence_length], name='Y')



# we first create a variable to take us from our one-hot representation to our LSTM cells
embedding = tf.get_variable("embedding", [n_chars, n_cells])

# And then use tensorflow's embedding lookup to look up the ids in X
Xs = tf.nn.embedding_lookup(embedding, X)

# The resulting lookups are concatenated into a dense tensor
print(Xs.get_shape().as_list())



# Let's create a name scope for the operations to clean things up in our graph#
with tf.name_scope('reslice'):
    Xs = [tf.squeeze(seq, [1]) for seq in tf.split(Xs, sequence_length, axis=1)]



cells = tf.contrib.rnn.BasicLSTMCell(num_units=n_cells)


initial_state = cells.zero_state(tf.shape(X)[0], tf.float32)




# Build deeper recurrent net if using more than 1 layer
if n_layers > 1:
    cells = [cells]
    for layer_i in range(1, n_layers):
        with tf.variable_scope('{}'.format(layer_i)):
            this_cell = tf.contrib.rnn.BasicLSTMCell(
                num_units=n_cells)
            cells.append(this_cell)
    cells = tf.contrib.rnn.MultiRNNCell(cells)
    initial_state = cells.zero_state(tf.shape(X)[0], tf.float32)



# this will return us a list of outputs of every element in our sequence.
# Each output is `batch_size` x `n_cells` of output.
# It will also return the state as a tuple of the n_cells's memory and
# their output to connect to the time we use the recurrent layer.
outputs, state = tf.contrib.rnn.static_rnn(cells, Xs, initial_state=initial_state)

# We'll now stack all our outputs for every cell
outputs_flat = tf.reshape(tf.concat(outputs, axis=1), [-1, n_cells])



with tf.variable_scope('prediction'):
    W = tf.get_variable(
        "W",
        shape=[n_cells, n_chars],
        initializer=tf.random_normal_initializer(stddev=0.1))
    b = tf.get_variable(
        "b",
        shape=[n_chars],
        initializer=tf.random_normal_initializer(stddev=0.1))

    # Find the output prediction of every single character in our minibatch
    # we denote the pre-activation prediction, logits.
    logits = tf.matmul(outputs_flat, W) + b

    # We get the probabilistic version by calculating the softmax of this
    probs = tf.nn.softmax(logits)

    # And then we can find the index of maximum probability
    Y_pred = tf.argmax(probs, axis=1)


with tf.variable_scope('loss'):
    # Compute mean cross entropy loss for each output.
    Y_true_flat = tf.reshape(tf.concat(Y, axis=1), [-1])
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=Y_true_flat)
    mean_loss = tf.reduce_mean(loss)


with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    gradients = []
    clip = tf.constant(5.0, name="clip")
    for grad, var in optimizer.compute_gradients(mean_loss):
        gradients.append((tf.clip_by_value(grad, -clip, clip), var))
    updates = optimizer.apply_gradients(gradients)



sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

cursor = 0
it_i = 0
while True:
    Xs, Ys = [], []
    for batch_i in range(batch_size):
        if (cursor + sequence_length) >= len(txt) - sequence_length - 1:
            cursor = 0
        Xs.append([encoder[ch]
                   for ch in txt[cursor:cursor + sequence_length]])
        Ys.append([encoder[ch]
                   for ch in txt[cursor + 1: cursor + sequence_length + 1]])

        cursor = (cursor + sequence_length)
    Xs = np.array(Xs).astype(np.int32)
    Ys = np.array(Ys).astype(np.int32)

    loss_val, _ = sess.run([mean_loss, updates],
                           feed_dict={X: Xs, Y: Ys})
    print(it_i, loss_val)

    if it_i % 500 == 0:
        p = sess.run([Y_pred], feed_dict={X: Xs})[0]
        preds = [decoder[p_i] for p_i in p]
        print("".join(preds).split('\n'))

    it_i += 1
