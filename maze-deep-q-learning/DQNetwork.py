import tensorflow as tf

class DQNetwork:
    """sets up the deque network"""

    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        """initializes the deque network"""
        
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            # create placeholders
            # state_size - size of each tuple: [None, 84, 84, 4]
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.placeholder(tf.float32, [None, 4], name="actions_")

            # target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.placeholder(tf.float32, [None], name="target")

            """
            First convnet:
            CNN
            BatchNormalization
            ELU
            """
            # Input is 84x84x4
            self.conv1 = tf.layers.conv2d(inputs = self.inputs_,
                                         filters = 64,
                                         kernel_size = [3,3],
                                         strides = [1,1],
                                         padding = "VALID",
                                         kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv1")

            self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm1')

            self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")
            ## --> [82, 82, 64]

            self.maxpool1 = tf.layers.max_pooling2d(inputs = self.conv1_out,
                                                    pool_size = [2,2],
                                                    strides = [2,2],
                                                    padding = "VALID",
                                                    name = "maxpool1")
            ## --> [41, 41, 64]

            """
            Second convnet:
            CNN
            BatchNormalization
            ELU
            """
            self.conv2 = tf.layers.conv2d(inputs = self.maxpool1,
                                 filters = 128,
                                 kernel_size = [3,3],
                                 strides = [1,1],
                                 padding = "VALID",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv2")

            self.conv2_batchnorm = tf.layers.batch_normalization(self.conv2,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm2')

            self.conv2_out = tf.nn.elu(self.conv2_batchnorm, name="conv2_out")
            ## --> [39, 39, 128]

            self.maxpool2 = tf.layers.max_pooling2d(inputs = self.conv2_out,
                                                    pool_size = [3,3],
                                                    strides = [2,2],
                                                    padding = "VALID",
                                                    name = "maxpool2")
            ## --> [19, 19, 128]
            """
            Third convnet:
            CNN
            BatchNormalization
            ELU
            """
            self.conv3 = tf.layers.conv2d(inputs = self.maxpool2,
                                 filters = 256,
                                 kernel_size = [3,3],
                                 strides = [1,1],
                                 padding = "VALID",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv3")

            self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm3')

            self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")
            ## --> [17, 17, 256]

            self.maxpool3 = tf.layers.max_pooling2d(inputs = self.conv3_out,
                                                    pool_size = [3,3],
                                                    strides = [2,2],
                                                    padding = "VALID",
                                                    name = "maxpool3")
            ## --> [8, 8, 256]


            """
            Fourth convnet:
            CNN
            BatchNormalization
            ELU
            """
            self.conv4 = tf.layers.conv2d(inputs = self.maxpool2,
                                 filters = 512,
                                 kernel_size = [3,3],
                                 strides = [1,1],
                                 padding = "VALID",
                                kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                 name = "conv4")

            self.conv4_batchnorm = tf.layers.batch_normalization(self.conv4,
                                                   training = True,
                                                   epsilon = 1e-5,
                                                     name = 'batch_norm4')

            self.conv4_out = tf.nn.elu(self.conv4_batchnorm, name="conv4_out")
            ## --> [6,6,512]

            self.maxpool4 = tf.layers.max_pooling2d(inputs = self.conv4_out,
                                                    pool_size = [2,2],
                                                    strides = [2,2],
                                                    padding = "VALID",
                                                    name = "maxpool4")
            ## --> [3, 3, 512]



            self.flatten = tf.layers.flatten(self.maxpool4)
            ## --> [4608]


            self.fc1 = tf.layers.dense(inputs = self.flatten,
                                  units = 1024,
                                  activation = tf.nn.elu,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                name="fc1")

            self.dropout1 = tf.layers.dropout(inputs=self.fc1, rate=0.3, training = True)

            self.fc2 = tf.layers.dense(inputs = self.dropout1,
                                 units = 512,
                                 activation = tf.nn.elu,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               name="fc2")

            self.dropout2 = tf.layers.dropout(inputs=self.fc2, rate=0.4, training = True)

            self.output = tf.layers.dense(inputs = self.dropout2,
                                           kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units = 4,
                                        activation=None)


            # Q is our predicted Q value.
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)


            # loss - difference between predicted Q_values and Q_target
            # Sum(Qtarget - Q)^2
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))

            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
