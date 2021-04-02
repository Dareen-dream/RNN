import FZ
import os
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import cv2

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 只显示error和warining信息

time_start = time.time()
FZ.fozu()

# './Users/mac/Downloads/mnist/'
# r'C:\Users\mac\Downloads\mnist.npz'
# mnist=input_data.read_data_sets("./2RNN/data",one_hot=True)
print("111")
mnist = input_data.read_data_sets(r'C:\Users\mac\Downloads\mnist.npz', one_hot=True)
print("222")
train_rate = 0.002
train_step = 1001
batch_size = 1000
display_step = 10

frame_size = 28
sequence_length = 28
hidden_num = 128
n_classes = 10

"""其中： 
train_rate是学习速率，这是一个超参数，目前由经验设置的，当然也可以自适应。

batch_size:每批样本数，rnn也可以使用随机梯度下降进行训练，一批批的灌数据进去，而不是每一次把整个数据集都灌进去。

sequence_size:每个样本序列的长度。因为我们希望把一个28x28的图片当做一个序列输入到rnn进行训练，所以我们需要对图片进行序列化。一种最方便的方法就是我们认为行与行之间存在某些关系，于是把图片的每一行取出来当做序列的一个维度。所以这里sequence_size就是设置为28。

反映到图1里面就是左边循环图展开后右图从左往右xi的数目。 rnn cell number

frame_size:序列里面每一个分量的大小。因为每个分量都是一行像素，而一行像素有28个像素点。所以frame_size为28。

反映到图1里面就是最下变的输入中每个xi都是一个长度为frame_size的向量或矩阵。input cell number

hidden_num：隐层个数，经验设置为5

反映到图1里面就是从下往上数的有hidden_num个隐层单元。

n_classes：类别数，10个数字就是设置为10咯"""

x = tf.placeholder(dtype=tf.float32, shape=[None, sequence_length*frame_size], name="inputx")

y = tf.placeholder(dtype=tf.float32,shape=[None, n_classes], name="expected_y")

weights = tf.Variable(tf.random_normal(shape=[hidden_num, n_classes]))
bias = tf.Variable(tf.fill([n_classes], 0.1))
"""注意：weights是整个网络的最后一层，它的形状为hidden_numXn_class,至于为什么是这个形状，我们下面来说。 
bias最后一层的偏置"""


# 定义RNN网络
def RNN(x, weights, bias):
    x = tf.reshape(x, shape=[-1, sequence_length, frame_size])
    # 先把输入转换为dynamic_rnn接受的形状：batch_size,sequence_length,frame_size这样子的
    # rnn_cell=tf.nn.rnn_cell.BasicRNNCell(hidden_num)
    rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_num)

    # 生成hidden_num个隐层的RNN网络,rnn_cell.output_size等于隐层个数，state_size也是等于隐层个数，但是对于LSTM单元来说这两个size又是不一样的。
    # 这是一个深度RNN网络,对于每一个长度为sequence_length的序列[x1,x2,x3,...,]的每一个xi,都会在深度方向跑一遍RNN,每一个都会被这hidden_num个隐层单元处理。

    output, states = tf.nn.dynamic_rnn(rnn_cell, x, dtype=tf.float32)
    h = tf.matmul(output[:, -1, :], weights)+bias
    # 此时output就是一个[batch_size,sequence_length,rnn_cell.output_size]形状的tensor
    return (h)
    # 我们取出最后每一个序列的最后一个分量的输出output[:,-1,:],它的形状为[batch_size,rnn_cell.output_size]也就是:[batch_size,hidden_num]所以它可以和weights相乘。这就是2.5中weights的形状初始化为[hidden_num,n_classes]的原因。然后再经softmax归一化。

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

    # saver.save(sess,'./2RNN/ckpt1/mnist1.ckpt',global_step=train_step)
    saver.save(sess, './model.ckpt', global_step=train_step)

time_end = time.time()
print('time cost', time_end-time_start, 's')