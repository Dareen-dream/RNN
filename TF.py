#coding:utf-8

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.datasets.mnist
# import tensorflow.compat.v1 as tf
# from tensorflow.examples.tutorials.mnist import input_data
# import tensorflow.examples.tutorials.mnist.input_data as input_data
tf.disable_v2_behavior()
# from tensorflow.keras.datasets.mnist import input_data
# from tensorflow.examples.tutorials.mnist import input_data
__all__ = [tf]

#init_state = tf.zeros(shape = [batch_size,rnn_cell.state_size])
#init_state = lstm_cell.zero_state(batch_size,dtype=tf.float32

##加载数据集，没有会自动下载的
# mnist = keras.datasets.mnist.load_data()
mnist = input_data.read_data_sets("./data",one_hot=True)

#常规参数
train_rate = 0.001
train_step = 10000
batch_size = 1280
display_step = 100

#rnn参数
frame_size = 28             #输入特征数
sequence_length = 28        #输入个数，时序
hidden_num = 100            #隐层神经元个数
n_classes = 10              #10个分类

#定义输入 定义输出
#此处输入格式是样本数*特征数，特征是把图片拉成一维的。None表示第一个维度可以是任意长度的
x=tf.placeholder(dtype=tf.float32,shape=[None,sequence_length*frame_size],name="inputx")
y=tf.placeholder(dtype=tf.float32,shape=[None,n_classes],name="expected_y")

#定义权值
#注意权值设定只设定 V   U和 W无需设定
weights=tf.Variable(tf.truncated_normal(shape=[hidden_num,n_classes]))    #全连接权重
bias = tf.Variable(tf.zeros(shape=[n_classes]))

def RNN(x,weights,bias):
    x=tf.reshape(x,shape=[-1,sequence_length,frame_size])          #3维
    rnn_cell = tf.nn.rnn_cell.BasicRNNCell(hidden_num)


    ###注意
    init_state = tf.zeros(shape=[batch_size,rnn_cell.state_size])#rnn_cell.state_size = 100
    init_state = rnn_cell.zero_state(batch_size,dtype=tf.float32)

    output,states=tf.nn.dynamic_rnn(rnn_cell,x,initial_state=init_state,dtype=tf.float32)
    return  tf.nn.softmax(tf.matmul(output[:,-1,:],weights)+bias,1)     #y=softmax(vh+1)

predy = RNN(x,weights,bias)

#以下所有神经网络大同小异
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predy,labels=y))
train = tf.train.AdamOptimizer(train_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(predy,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.to_float(correct_pred))

sess = tf.Session()
sess.run(tf.global_variables_initializer())
step = 1
testx,testy = mnist.test.next_batch(batch_size)
saver = tf.train.Saver()
while step<train_step:
    batch_x,batch_y = mnist.train.next_batch(batch_size)
    _loss,__=sess.run([cost,train],feed_dict={x:batch_x,y:batch_y})
    if step % display_step == 0:
        print()
        acc,loss = sess.run([accuracy,cost],feed_dict={x:testx,y:testy})
        print(step,acc,loss)
    step+=1