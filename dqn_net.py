import torch
import torch.nn as nn
import tensorflow._api.v2.compat.v1 as tf
## hidden_layer_size = 20

class QNetworktest():
    def __init__(self, state_input,action_dim):
        self.state = state_input
        self.action_space = action_dim

        W_conv1 = self.weight_variable([5,5,1,32])
        b_conv1 = self.bias_variable([32])
        W_conv2 = self.weight_variable([5,5,32,64])
        b_conv2 = self.bias_variable([64])

        W1 = self.weight_variable([int((1280/4) * (720/4) * 64), 512])
        b1 = self.bias_variable([512])

        W2 = self.weight_variable([512, 256])
        b2 = self.bias_variable([256])

        W3 = self.weight_variable([256, action_dim])
        b3 = self.bias_variable([action_dim])

        h_conv1 = tf.nn.relu(self.conv2d(state_input, W_conv1) + b_conv1)   
        h_pool1 = self.max_pool_2x2(h_conv1) 
        
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2) 
        h_pool2 = self.max_pool_2x2(h_conv2) 

        h_conv2_flat = tf.reshape(h_pool2, [-1,int((1280/4) * (720/4) * 64)])
        h_layer_one = tf.nn.relu(tf.matmul(h_conv2_flat, W1) + b1)

        h_layer_one = tf.nn.dropout(h_layer_one, 1)

        h_layer_two = tf.nn.relu(tf.matmul(h_layer_one, W2) + b2)

        h_layer_two = tf.nn.dropout(h_layer_two, 1)

        Q_value = tf.matmul(h_layer_two, W3) + b3
        Q_value = tf.nn.dropout(Q_value, 1)
    def create_updating_method(self):
        # this the input action, use one hot presentation
        self.action_input = tf.placeholder("float", [None, self.action_dim])
        # this the TD aim value
        self.y_input = tf.placeholder("float", [None])
        # this the action's Q_value
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), reduction_indices=1)
        # 生成的Q_value实际上是一个action大小的list,action_input是一个one-hot向量,
        # 两者相乘实际上是取出了执行操作的Q值进行单独更新
        # this is the lost
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        # 均方差损失函数
        # drawing loss graph
        tf.summary.scalar('loss',self.cost)
        # loss graph save
        with tf.name_scope('train_loss'):
            # use the loss to optimize the network
            self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.cost)
            # learning_rate=0.0001
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)  

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)
    
    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')
    
    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')