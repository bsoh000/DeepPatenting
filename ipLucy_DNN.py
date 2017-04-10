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
filename = '003'

############
# read data in np array
############
xy = np.loadtxt('./data/data_'+filename+'.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, :n_feat]
y = xy[:, -1:]
y = y.astype(int)
y = y.reshape([-1])

y_data = np.eye(nb_classes)[y]

# y_data = tf.reshape(y_one_hot, [-1, nb_classes])

print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)

'''

############
# read data in queue
############
filename_queue = tf.train.string_input_producer(
    # ['./data/data_001.csv',
    #  './data/data_002.csv',
    #  './data/data_003.csv',
    #  './data/data_004.csv',
     ['./data/data_005.csv'],
    shuffle=True, name='filename_queue')

reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# Default values, in case of empty columns. Also specifies the type of the
# decoded result.
# record_defaults = [[0.], [0.], [0.], [0.]]
record_defaults = [[0.], [0.], [0.], [0.], [0.],
                   [0.], [0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)

# collect batches of csv in
batch_size = 10
train_x_batch, Y = tf.train.batch([xy[0:n_feat], xy[-1:]], batch_size=batch_size)

# min_after_dequeue = 10000
# capacity = min_after_dequeue + 3 * batch_size
# train_x_batch, Y = tf.train.shuffle_batch([xy[0:n_feat], xy[-1:]],
#                                                       batch_size=batch_size,
#                                                       capacity=capacity,
#                                                       min_after_dequeue=min_after_dequeue)

Y = tf.cast(Y, tf.int32)
Y_one_hot = tf.one_hot(Y, nb_classes)  # one hot
print("one_hot", Y_one_hot)
train_y_batch = tf.reshape(Y_one_hot, [-1, nb_classes])
print("one_hot", train_y_batch)

'''

#############
# NN model
#############
# input layer
X = tf.placeholder(tf.float32)  # for data
Y = tf.placeholder(tf.int32)  # for label

# 3 layers of hidden layers
# 10 => n_sym+1 (input) -> 32 (hidden 1) -> 128 (hidden 2) -> 256 (hidden 3) -> nb_classes (output)
W1 = tf.get_variable("W1", shape=[n_feat, 32], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.get_variable("b1", shape=[32], initializer=tf.contrib.layers.xavier_initializer())

W2 = tf.get_variable("W2", shape=[32, 64], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.get_variable("b2", shape=[64], initializer=tf.contrib.layers.xavier_initializer())

W3 = tf.get_variable("W3", shape=[64, 128], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.get_variable("b3", shape=[128], initializer=tf.contrib.layers.xavier_initializer())

W4 = tf.get_variable("W4", shape=[128, nb_classes], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.get_variable("b4", shape=[nb_classes], initializer=tf.contrib.layers.xavier_initializer())

# hidden layers
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)

# output
# if cost = softmax_cross_entropy_with_logits,
# then no need to add softmax to output
model = tf.matmul(L3, W4) + b4
hypothesis = tf.nn.softmax(model)

learning_rate = 0.001
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
# optimizer = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(cost)
tf.add_to_collection('train_op', train_op)

############
# read test data in np array
############
test_xy = np.loadtxt('./data/test_data.csv', delimiter=',', dtype=np.float32)
test_x_data = test_xy[:, :n_feat]
ty = test_xy[:, -1:]

ty = ty.astype(int)
ty = ty.reshape([-1])

# nb_classes = 3
test_y_data = np.eye(nb_classes)[ty]

#############
# test variables
#############
# raw_predict = hypothesis
raw_predict = model
predict = tf.argmax(raw_predict, 1)
target = tf.argmax(Y, 1)
check_predict = tf.equal(predict, target)
accuracy = tf.reduce_mean(tf.cast(check_predict, tf.float32))*100

#############
# Summaries
#############
# summary scalar for cost
cost_sum = tf.summary.scalar('cost', cost)
acc_sum = tf.summary.scalar('accuracy', accuracy)

mean1 = tf.reduce_mean(W1)
stddev1 = tf.sqrt(tf.reduce_mean(tf.square(W1 - mean1)))
mean1_sum = tf.summary.scalar('mean1', mean1)
std1_sum = tf.summary.scalar('stddev1', stddev1)
max1_sum = tf.summary.scalar('max1', tf.reduce_max(W1))
min1_sum = tf.summary.scalar('min1', tf.reduce_min(W1))

# histograms
w1h = tf.summary.histogram('weight1', W1)
b1h = tf.summary.histogram('bias', b1)
w2h = tf.summary.histogram('weight2', W2)
b2h = tf.summary.histogram('bias2', b2)
w3h = tf.summary.histogram('weight3', W3)
b3h = tf.summary.histogram('bias3', b3)
w4h = tf.summary.histogram('weight4', W4)
b4h = tf.summary.histogram('bias4', b4)
mh = tf.summary.histogram('model', model)

# merge summaries
merged = tf.summary.merge([cost_sum, std1_sum, mh])

#############
# learning
#############
init = tf.global_variables_initializer()

with tf.Session() as sess:
    # sess.run(init)    # deactivate when restoring
    saver = tf.train.Saver()
    model_name = "./ckpt/DNN/deep"

    # for restoring pre-trained ckpt
    saver.restore(sess, model_name)
    train_op = tf.get_collection('train_op')[0]

    b4ini = sess.run(b4)

    # queue coordinator
    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # save summary
    train_writer = tf.summary.FileWriter('./logs/DNN/train/'+filename, sess.graph)
    test_writer = tf.summary.FileWriter('./logs/DNN/test/'+filename, sess.graph)

    used_x, used_y = [],[]
    t_used_x, t_used_y = [],[]

    for step in range(100000):
        # for batch extraction
        # x_data, y_data = sess.run([train_x_batch, train_y_batch])

        # for train
        sess.run(train_op, feed_dict={X:x_data, Y:y_data})
        # for confirming used training data
        used_x.append(x_data)
        used_y.append(y_data)

        accuracy_val = sess.run(accuracy, feed_dict={X: test_x_data, Y: test_y_data})
        # for confirming used test data
        t_used_x.append(test_x_data)
        t_used_y.append(test_y_data)

        # for test
        # raw_predict_val = sess.run(raw_predict, feed_dict={X: test_x_data}) * 100
        # predict_val = sess.run(predict, feed_dict={X: test_x_data})
        # target_val = sess.run(target, feed_dict={Y: test_y_data})

        if (step + 1) % 100 == 0:
            cost_val = sess.run(cost, feed_dict={X: x_data, Y: y_data})
            print(step+1, 'cost =', cost_val, 'acc =', accuracy_val)

        summary = sess.run(merged, feed_dict={X: x_data, Y: y_data})
        train_writer.add_summary(summary, step)

        test_summary = sess.run(acc_sum, feed_dict={X: test_x_data, Y: test_y_data})
        test_writer.add_summary(test_summary, step)

    # closing queue coordinator
    # coord.request_stop()
    # coord.join(threads)

    # run tensorboard: terminal outside python tensorboard --logdir=logs/DNN
    # quit tensorboard: ^+c

    # saving used data
    used_data = np.c_[used_x, used_y]
    # np.savetxt('./data/DNN_used_data'+filename+'.csv', used_data, delimiter=",")

    t_used_data = np.c_[t_used_x, t_used_y]
    # np.savetxt('./data/DNN_t_used_data'+filename+'.csv', t_used_data, delimiter=",")

    # Save model values
    save_path = saver.save(sess, model_name)
    print('save_path: ', save_path, '\n')
    print("b4ini: ", b4ini)
    print("b4: ", sess.run(b4))

# check the result
# 0: 102, 1: 103, 2: non-similar
print("DNN Training Time:", (time.time()-t))
