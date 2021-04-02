import FZ
import os
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 只显示error和warining信息

time_start = time.time()
FZ.fozu()

mnist = input_data.read_data_sets(r'C:\Users\mac\Downloads\mnist.npz', one_hot=True)

train_rate = 0.002
train_step = 1001
batch_size = 1000
display_step = 10

frame_size = 28
sequence_length = 28
hidden_num = 128
n_classes = 10

x = tf.placeholder(dtype=tf.float32, shape=[None, sequence_length*frame_size], name="inputx")

y = tf.placeholder(dtype=tf.float32,shape=[None, n_classes], name="expected_y")

weights = tf.Variable(tf.random_normal(shape=[hidden_num, n_classes]))
bias = tf.Variable(tf.fill([n_classes], 0.1))

# 定义RNN网络
def RNN(x, weights, bias):
    x = tf.reshape(x, shape=[-1, sequence_length, frame_size])

    rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_num)

    output, states = tf.nn.dynamic_rnn(rnn_cell, x, dtype=tf.float32)
    h = tf.matmul(output[:, -1, :], weights)+bias

    return (h)

predy = RNN(x, weights, bias)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predy,labels=y))

opt = tf.train.AdamOptimizer(train_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(predy, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.to_float(correct_pred))

testx, testy = mnist.test.next_batch(batch_size)

saver = tf.train.Saver()

with tf.Session() as sess:
    srun = sess.run
    init = tf.global_variables_initializer()
    srun(init)
    for t in range(train_step):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        _cost_val, _ = srun([cost, opt], {x: batch_x, y: batch_y})
        if(t % display_step == 0):
            accuracy_val, cost_val = srun([accuracy, cost], {x: testx, y: testy})
            print(t, cost_val, accuracy_val)

    saver.save(sess, './model.ckpt', global_step=train_step)

time_end = time.time()
print('time cost', time_end-time_start, 's')