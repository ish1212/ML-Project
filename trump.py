import sys
import os
import numpy as np

import tensorflow as tf



with open('trump.txt', 'r', encoding='utf-8') as fp:
    txt = fp.read()


tf.reset_default_graph()



txt = "\n".join([txt_i.strip()
                 for txt_i in txt.replace('\t', '').split('\n')
                 if len(txt_i)])

'''
import re

strip_special_chars  = re.compile("[^A-Za-z0-9 ]+")

def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())


text=""
for i, char in enumerate(txt):
    text+=cleanSentences(char)
'''

vocab = list(set(txt))
len(txt), len(vocab)

from collections import OrderedDict

encoder = OrderedDict(zip(vocab, range(len(vocab))))
decoder = OrderedDict(zip(range(len(vocab)), vocab))


words = set(txt.split(' '))

counts = {word_i: 0 for word_i in words}
for word_i in txt.split(' '):
    counts[word_i] += 1


[(word_i, counts[word_i]) for word_i in sorted(counts, key=counts.get, reverse=True)]



from libs import charrnn


ckpt_name = './pretrained_lstm.ckpt' #'./trump.ckpt'
g = tf.Graph()
n_layers = 2
n_cells = 512
with tf.Session(graph=g) as sess:
    model = charrnn.build_model(txt=txt,
                                batch_size=1,
                                sequence_length=1,
                                n_layers=n_layers,
                                n_cells=n_cells,
                                gradient_clip=10.0)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    if os.path.exists(ckpt_name):
        # saver = tf.train.import_meta_graph(ckpt_name)
        # saver.restore(sess,tf.train.latest_checkpoint('models'))
        saver.restore(sess, ckpt_name)
        print("Model restored.")
    else:
        print("wtf?")



n_iterations = 500

# save
# # Inference: Keeping Track of the State
#
# curr_states  = None
# g = tf.Graph()
# with tf.Session(graph=g) as sess:
#     model = charrnn.build_model(txt=txt,
#                                 batch_size=1,
#                                 sequence_length=1,
#                                 n_layers=n_layers,
#                                 n_cells=n_cells,
#                                 gradient_clip=10.0)
#     sess.run(tf.global_variables_initializer())
#     saver = tf.train.Saver()
#     if os.path.exists(ckpt_name):
#         saver.restore(sess, ckpt_name)
#         print("Model restored.")
#
#     # Get every tf.Tensor for the initial state
#     init_states = []
#     for s_i in model['initial_state']:
#         init_states.append(s_i.c)
#         init_states.append(s_i.h)
#
#     # Similarly, for every state after inference
#     final_states = []
#     for s_i in model['final_state']:
#         final_states.append(s_i.c)
#         final_states.append(s_i.h)
#
#     # Let's start with the letter 't' and see what comes out:
#     synth = [[encoder[' ']]]
#     for i in range(n_iterations):
#
#         # We'll create a feed_dict parameter which includes what to
#         # input to the network, model['X'], as well as setting
#         # dropout to 1.0, meaning no dropout.
#         feed_dict = {model['X']: [synth[-1]],
#                      model['keep_prob']: 1.0}
#
#         # Now we'll check if we currently have a state as a result
#         # of a previous inference, and if so, add to our feed_dict
#         # parameter the mapping of the init_state to the previous
#         # output state stored in "curr_states".
#         if curr_states:
#             feed_dict.update(
#                 {init_state_i: curr_state_i
#                  for (init_state_i, curr_state_i) in
#                      zip(init_states, curr_states)})
#
#         # Now we can infer and see what letter we get
#         p = sess.run(model['probs'], feed_dict=feed_dict)[0]
#
#         # And make sure we also keep track of the new state
#         curr_states = sess.run(final_states, feed_dict=feed_dict)
#
#         # Find the most likely character
#         p = np.argmax(p)
#
#         # Append to string
#         synth.append([p])
#
#         # Print out the decoded letter
#         print(model['decoder'][p], end='')
#         sys.stdout.flush()
#
#
# # Probabilistic Sampling¶
#
# curr_statescurr_sta  = None
# g = tf.Graph()
# with tf.Session(graph=g) as sess:
#     model = charrnn.build_model(txt=txt,
#                                 batch_size=1,
#                                 sequence_length=1,
#                                 n_layers=n_layers,
#                                 n_cells=n_cells,
#                                 gradient_clip=10.0)
#     sess.run(tf.global_variables_initializer())
#     saver = tf.train.Saver()
#     if os.path.exists(ckpt_name):
#         saver.restore(sess, ckpt_name)
#         print("Model restored.")
#
#     # Get every tf.Tensor for the initial state
#     init_states = []
#     for s_i in model['initial_state']:
#         init_states.append(s_i.c)
#         init_states.append(s_i.h)
#
#     # Similarly, for every state after inference
#     final_states = []
#     for s_i in model['final_state']:
#         final_states.append(s_i.c)
#         final_states.append(s_i.h)
#
#     # Let's start with the letter 't' and see what comes out:
#     synth = [[encoder[' ']]]
#     for i in range(n_iterations):
#
#         # We'll create a feed_dict parameter which includes what to
#         # input to the network, model['X'], as well as setting
#         # dropout to 1.0, meaning no dropout.
#         feed_dict = {model['X']: [synth[-1]],
#                      model['keep_prob']: 1.0}
#
#         # Now we'll check if we currently have a state as a result
#         # of a previous inference, and if so, add to our feed_dict
#         # parameter the mapping of the init_state to the previous
#         # output state stored in "curr_states".
#         if curr_states:
#             feed_dict.update(
#                 {init_state_i: curr_state_i
#                  for (init_state_i, curr_state_i) in
#                      zip(init_states, curr_states)})
#
#         # Now we can infer and see what letter we get
#         p = sess.run(model['probs'], feed_dict=feed_dict)[0]
#
#         # And make sure we also keep track of the new state
#         curr_states = sess.run(final_states, feed_dict=feed_dict)
#
#         # Now instead of finding the most likely character,
#         # we'll sample with the probabilities of each letter
#         p = p.astype(np.float64)
#         p = np.random.multinomial(1, p.ravel() / p.sum())
#         p = np.argmax(p)
#
#         # Append to string
#         synth.append([p])
#
#         # Print out the decoded letter
#         print(model['decoder'][p], end='')
#         sys.stdout.flush()
#
#
# # Inference: Temperature
# temperature = 0.5
# curr_states = None
# g = tf.Graph()
# with tf.Session(graph=g) as sess:
#     model = charrnn.build_model(txt=txt,
#                                 batch_size=1,
#                                 sequence_length=1,
#                                 n_layers=n_layers,
#                                 n_cells=n_cells,
#                                 gradient_clip=10.0)
#     sess.run(tf.global_variables_initializer())
#     saver = tf.train.Saver()
#     if os.path.exists(ckpt_name):
#         saver.restore(sess, ckpt_name)
#         print("Model restored.")
#
#     # Get every tf.Tensor for the initial state
#     init_states = []
#     for s_i in model['initial_state']:
#         init_states.append(s_i.c)
#         init_states.append(s_i.h)
#
#     # Similarly, for every state after inference
#     final_states = []
#     for s_i in model['final_state']:
#         final_states.append(s_i.c)
#         final_states.append(s_i.h)
#
#     # Let's start with the letter 't' and see what comes out:
#     synth = [[encoder[' ']]]
#     for i in range(n_iterations):
#
#         # We'll create a feed_dict parameter which includes what to
#         # input to the network, model['X'], as well as setting
#         # dropout to 1.0, meaning no dropout.
#         feed_dict = {model['X']: [synth[-1]],
#                      model['keep_prob']: 1.0}
#
#         # Now we'll check if we currently have a state as a result
#         # of a previous inference, and if so, add to our feed_dict
#         # parameter the mapping of the init_state to the previous
#         # output state stored in "curr_states".
#         if curr_states:
#             feed_dict.update(
#                 {init_state_i: curr_state_i
#                  for (init_state_i, curr_state_i) in
#                      zip(init_states, curr_states)})
#
#         # Now we can infer and see what letter we get
#         p = sess.run(model['probs'], feed_dict=feed_dict)[0]
#
#         # And make sure we also keep track of the new state
#         curr_states = sess.run(final_states, feed_dict=feed_dict)
#
#         # Now instead of finding the most likely character,
#         # we'll sample with the probabilities of each letter
#         p = p.astype(np.float64)
#         p = np.log(p) / temperature
#         p = np.exp(p) / np.sum(np.exp(p))
#         p = np.random.multinomial(1, p.ravel() / p.sum())
#         p = np.argmax(p)
#
#         # Append to string
#         synth.append([p])
#
#         # Print out the decoded letter
#         print(model['decoder'][p], end='')
#         sys.stdout.flush()
#
#
# # Inference: Priming
#
# prime = "obama"
# temperature = 1.0
# curr_states = None
# n_iterations = 500
# g = tf.Graph()
# with tf.Session(graph=g) as sess:
#     model = charrnn.build_model(txt=txt,
#                                 batch_size=1,
#                                 sequence_length=1,
#                                 n_layers=n_layers,
#                                 n_cells=n_cells,
#                                 gradient_clip=10.0)
#     sess.run(tf.global_variables_initializer())
#     saver = tf.train.Saver()
#     if os.path.exists(ckpt_name):
#         saver.restore(sess, ckpt_name)
#         print("Model restored.")
#
#     # Get every tf.Tensor for the initial state
#     init_states = []
#     for s_i in model['initial_state']:
#         init_states.append(s_i.c)
#         init_states.append(s_i.h)
#
#     # Similarly, for every state after inference
#     final_states = []
#     for s_i in model['final_state']:
#         final_states.append(s_i.c)
#         final_states.append(s_i.h)
#
#     # Now we'll keep track of the state as we feed it one
#     # letter at a time.
#     curr_states = None
#     for ch in prime:
#         feed_dict = {model['X']: [[model['encoder'][ch]]],
#                      model['keep_prob']: 1.0}
#         if curr_states:
#             feed_dict.update(
#                 {init_state_i: curr_state_i
#                  for (init_state_i, curr_state_i) in
#                      zip(init_states, curr_states)})
#
#         # Now we can infer and see what letter we get
#         p = sess.run(model['probs'], feed_dict=feed_dict)[0]
#         p = p.astype(np.float64)
#         p = np.log(p) / temperature
#         p = np.exp(p) / np.sum(np.exp(p))
#         p = np.random.multinomial(1, p.ravel() / p.sum())
#         p = np.argmax(p)
#
#         # And make sure we also keep track of the new state
#         curr_states = sess.run(final_states, feed_dict=feed_dict)
#
#     # Now we're ready to do what we were doing before but with the
#     # last predicted output stored in `p`, and the current state of
#     # the model.
#     synth = [[p]]
#     print(prime + model['decoder'][p], end='')
#     for i in range(n_iterations):
#
#         # Input to the network
#         feed_dict = {model['X']: [synth[-1]],
#                      model['keep_prob']: 1.0}
#
#         # Also feed our current state
#         feed_dict.update(
#             {init_state_i: curr_state_i
#              for (init_state_i, curr_state_i) in
#                  zip(init_states, curr_states)})
#
#         # Inference
#         p = sess.run(model['probs'], feed_dict=feed_dict)[0]
#
#         # Keep track of the new state
#         curr_states = sess.run(final_states, feed_dict=feed_dict)
#
#         # Sample
#         p = p.astype(np.float64)
#         p = np.log(p) / temperature
#         p = np.exp(p) / np.sum(np.exp(p))
#         p = np.random.multinomial(1, p.ravel() / p.sum())
#         p = np.argmax(p)
#
#         # Append to string
#         synth.append([p])
#
#         # Print out the decoded letter
#         print(model['decoder'][p], end='')
#         sys.stdout.flush()
