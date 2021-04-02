import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 载入数据集
mnist = input_data.read_data_sets("MNIST_DATA/", one_hot=True)

# 构建网络模型
n_inputs = 28  # 输入一行，一行28个
max_time = 28  # 一共28行
lstm_size = 100  # 隐层单元
n_classes = 10  # 10个分类
batch_size = 64  # 每批次64个样本
n_batch = mnist.train.num_examples // batch_size
x = tf.placeholder(tf.float32, [None, 784], name='x-input')
y = tf.placeholder(tf.float32, [None, 10], name='y-input')
# 初始化权值 输出层
weights = tf.Variable(tf.truncated_normal([lstm_size, n_classes], stddev=0.1))
biases = tf.Variable(tf.constant(0.1, shape=[n_classes]))


# 定义RNN网络
def RNN(X, weights, biases):
    inputs = tf.reshape(X, [-1, max_time, n_inputs])
    # 定义LSTM
    lstm_cell = tf.nn.rnn_cell.LSTMCell(lstm_size)
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)  # 最后一行传进去时的输出结果
    results = tf.nn.softmax(tf.matmul(final_state[1], weights) + biases)
    return results


prediction = RNN(x, weights, biases)
loss = tf.losses.softmax_cross_entropy(y, prediction)
train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(prediction, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(11):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter " + str(epoch) + ", Testing Accuracy= " + str(acc))
