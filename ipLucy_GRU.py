# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import time
t = time.time()

############
# data info
############
n_sym = 8
n_feat = n_sym + 1
nb_classes = 3
filename = '005'

n_input = 1         # len(x_data[0])
n_steps = n_feat
n_classes = nb_classes
batch_size = 10
n_hidden = 128
n_layers = 5
grad_clip = 5.


############
# read data in np array
############
xy = np.loadtxt('./data/data_'+filename+'.csv', delimiter=',', dtype=np.float32)
x = xy[:, :n_feat]
y = xy[:, -1:]

x_data = x.reshape([-1, n_steps, n_input])
y_data = y.reshape([-1, 1])

print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)

'''
############
# read data in queue
############
filename_queue = tf.train.string_input_producer(
    ['./data/data_001.csv',
     './data/data_002.csv',
     './data/data_003.csv',
     './data/data_004.csv',
     './data/data_005.csv'],
    shuffle=False, name='filename_queue')

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
record_defaults = [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)

# collect batches of csv in
# batch_size = 100
# train_x_batch, train_y_batch = tf.train.batch([xy[0:10], xy[10:]], batch_size=batch_size)

min_after_dequeue = 10000
capacity = min_after_dequeue + 3 * batch_size
train_x_batch, train_y_batch = tf.train.shuffle_batch([xy[0:n_feat], xy[-1:]],
                                                      batch_size=batch_size,
                                                      capacity=capacity,
                                                      min_after_dequeue=min_after_dequeue)

'''

#############
# NN model
#############
# input layer
# [batch size, time steps, input size]
X = tf.placeholder(tf.float32, [None, n_steps, n_input])  # for data
# [batch_size, n_steps, n_input] -> Tensor[n_steps, batch_size, n_input]
X_t = tf.transpose(X, [1, 0, 2])
# X_t = tf.reshape(X_t, [-1, n_input])

# [batch size, time steps]
Y = tf.placeholder(tf.int64, [None, 1])  # for label

# Weight & Bias
W = tf.get_variable("W", shape=[n_hidden, n_classes], initializer=tf.contrib.layers.xavier_initializer())
b = tf.get_variable("b", shape=[n_classes], initializer=tf.contrib.layers.xavier_initializer())

# RNN & Dropout
cell = tf.contrib.rnn.GRUCell(num_units=n_hidden)
cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.5)

# multi-layer RNN
cell = tf.contrib.rnn.MultiRNNCell([cell] * n_layers)

# Dynamic RNN
outputs, states = tf.nn.dynamic_rnn(cell, X_t, dtype=tf.float32, time_major=True)

# logits of shape `[batch_size, num_classes]` and labels of shape `[batch_size]`.
logits = tf.matmul(outputs[-1], W) + b
hypothesis = tf.nn.softmax(logits)

# if cost = softmax_cross_entropy_with_logits,
# then no need to add softmax to output
labels = tf.reshape(Y, [-1])

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
# optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9)
train_op = optimizer.minimize(cost)
tf.add_to_collection('train_op', train_op)

############
# read test data in np array
############
test_xy = np.loadtxt('./data/test_data.csv', delimiter=',', dtype=np.float32)
test_x_data = test_xy[:, :n_feat]
test_y_data = test_xy[:, -1:]

test_x_data = test_x_data.reshape([-1, n_steps, n_input])
test_y_data = test_y_data.reshape([-1, 1])


#############
# test variables
#############
raw_predict = hypothesis
predict = tf.argmax(raw_predict, 1)
target = labels
check_predict = tf.equal(predict, target)
accuracy = tf.reduce_mean(tf.cast(check_predict, tf.float32))*100

#############
# Summaries
#############
# summary scalar for cost
cost_sum = tf.summary.scalar('cost', cost)
acc_sum = tf.summary.scalar('accuracy', accuracy)

mean = tf.reduce_mean(W)
stddev = tf.sqrt(tf.reduce_mean(tf.square(W - mean)))
mean_sum = tf.summary.scalar('mean', mean)
std_sum = tf.summary.scalar('stddev', stddev)
max_sum = tf.summary.scalar('max', tf.reduce_max(W))
min_sum = tf.summary.scalar('min', tf.reduce_min(W))

# histograms
wh = tf.summary.histogram('weight', W)
bh = tf.summary.histogram('bias', b)
lh = tf.summary.histogram('logits', logits)

# merge summaries
merged = tf.summary.merge([cost_sum, std_sum, lh])

#############
# learning
#############
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # queue coordinator
    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    sess.run(init)    # need for the first time training
    saver = tf.train.Saver()
    model_name = "./ckpt/GRU/deep"
    # saver.restore(sess, model_name)

    # training
    # train_op = tf.get_collection('train_op')[0]
    bini = sess.run(b)

    # testing

    # save summary
    train_writer = tf.summary.FileWriter('./logs/GRU/train', sess.graph)
    test_writer = tf.summary.FileWriter('./logs/GRU/test', sess.graph)

    for step in range(15000):
        # for batch
        # x_data, y_data = sess.run([train_x_batch, train_y_batch])
        # x_data = x_data.reshape([-1, n_steps, n_input])
        # y_data = y_data.reshape([-1, 1])

        # for train
        sess.run(train_op, feed_dict={X: x_data, Y: y_data})
        accuracy_val = sess.run(accuracy, feed_dict={X: test_x_data, Y: test_y_data})

        if (step + 1) % 100 == 0:
            cost_val = sess.run(cost, feed_dict={X: x_data, Y: y_data})
            # for test
            # raw_predict_val = sess.run(raw_predict, feed_dict={X: test_x_data}) * 100
            # predict_val = sess.run(predict, feed_dict={X: test_x_data})
            # target_val = sess.run(target, feed_dict={Y: test_y_data})
            print(step+1, 'cost =', cost_val, 'acc =', accuracy_val)

        summary = sess.run(merged, feed_dict={X:x_data, Y:y_data})
        train_writer.add_summary(summary, step)

        test_summary = sess.run(acc_sum, feed_dict={X: test_x_data, Y: test_y_data})
        test_writer.add_summary(test_summary, step)
        # run tensorboard: python -m tensorflow.tensorboard --logdir=logs/GRU
        # quit tensorboard: ^+c

    # coord.request_stop()
    # coord.join(threads)

    # Save model values
    save_path = saver.save(sess, './ckpt/GRU/deep')
    print('save_path: ', save_path, '\n')
    print("bini: ", bini)
    print("b: ", sess.run(b))

print("Actual: ", target)
# check the result
# 0: self, 1: 102, 2: 103, 3: non-art

print("GRU Training Time:", (time.time()-t))