import tensorflow as tf


def lrelu(x,alpha=0.2):
    x = tf.maximum(alpha * x, x)
    #x = x * tf.nn.sigmoid(x)
    return x


def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat([x , y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2] , y_shapes[3]])], 3)   #对于一个二维矩阵，第0个维度代表最外层方括号所框下的子集，第1个维度代表内部方括号所框下的子集。维度越高，括号越小

def create_variables(name, shape, initializer=tf.truncated_normal_initializer(stddev=0.02), is_fc_layer=False):    #生成截断正态分布的初始化程序  stddev：一个 python 标量或一个标量张量,要生成的随机值的标准偏差

    new_variables = tf.get_variable(name, shape=shape, initializer=initializer)
    return new_variables

def output_layer(input_layer, num_labels):
    '''
    :param input_layer: 2D tensor
    :param num_labels: int. How many output labels in total? (10 for cifar10 and 100 for cifar100)
    :return: output layer Y = WX + B
    '''
    input_dim = input_layer.get_shape().as_list()[-1]  #tensor.get_shape()返回的是元组，不能放到sess.run()里面，这个里面只能放operation和tensor
    fc_w = create_variables(name='fc_weights', shape=[input_dim, num_labels], is_fc_layer=True,
                            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))          #初始化器,可生成张量而不会缩放方差
    fc_b = create_variables(name='fc_bias', shape=[num_labels], initializer=tf.zeros_initializer())

    fc_h = tf.matmul(input_layer, fc_w) + fc_b
    return fc_h

def conv_bn_relu_layer(input_layer, filter_shape, stride,is_training):
    '''
    A helper function to conv, batch normalize and relu the input tensor sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
    '''

    filter = create_variables(name='conv', shape=filter_shape)

    conv_layer = tf.nn.conv2d(input_layer, filter, strides=[1, stride, stride, 1], padding='SAME')     #padding='SAME' 输出大小等于输入大小除以步长向上取整，s是步长大小
    # conv_layer = tf.layers.batch_normalization(conv_layer, training=is_training)
    # output = lrelu(conv_layer)
    output = tf.nn.leaky_relu(conv_layer, alpha= 0.8)
    return output


def bn_relu_conv_layer(input_layer, filter_shape, stride,is_training):
    '''
    A helper function to batch normalize, relu and conv the input layer sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = conv(Relu(batch_normalize(X)))
    '''
    relu_layer = lrelu(input_layer)

    filter = create_variables(name='conv', shape=filter_shape)
    bias = create_variables(name='bias', shape=filter_shape[-1])
    conv_layer = tf.nn.conv2d(relu_layer, filter, strides=[1, stride, stride, 1], padding='SAME')
    conv_layer = tf.layers.batch_normalization(conv_layer, training=is_training)
    # return conv_layer
    return conv_layer+bias



def residual_block(input_layer, output_channel, is_training,first_block=False):
    '''
    Defines a residual block in ResNet
    :param input_layer: 4D tensor
    :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
    :param first_block: if this is the first residual block of the whole network
    :return: 4D tensor.
    '''
    input_channel = input_layer.get_shape().as_list()[-1]

    # When it's time to "shrink" the image size, we use stride = 2
    if input_channel * 2 == output_channel:
        increase_dim = True
        stride = 2
    elif input_channel == output_channel:
        increase_dim = False
        stride = 1
    else:
        raise ValueError('Output and input channel does not match in residual blocks!!!')

    # The first conv layer of the first residual block does not need to be normalized and relu-ed.
    with tf.variable_scope('conv1_in_block'):
        if first_block:
            filter = create_variables(name='conv', shape=[3, 3, input_channel, output_channel])
            bias = create_variables(name='bias', shape=[output_channel])
            conv1 = tf.nn.conv2d(input_layer, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
            # conv1 = tf.layers.batch_normalization(conv1, training=is_training)
            # conv1 = conv1+bias
        else:
            conv1 = bn_relu_conv_layer(input_layer, [3, 3, input_channel, output_channel], stride,is_training)

    with tf.variable_scope('conv2_in_block'):
        conv2 = bn_relu_conv_layer(conv1, [3, 3, output_channel, output_channel], 1,is_training)

    # When the channels of input layer and conv2 does not match, we add zero pads to increase the
    #  depth of input layers
    if increase_dim is True:
        pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1], padding='SAME')   #padding = “VALID”输入和输出大小关系如下  输出大小等于输入大小减去滤波器大小加上1，最后再除以步长
        padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                     input_channel // 2]])       #paddings=[a,b,c,d],分别从不同维度加0
    else:
        padded_input = input_layer

    output = conv2 + padded_input
    return output

def inv_relu_conv_layer(input_layer, input_channel,output_channel, filter_size,out_size,batch_size,stride,is_training):
    '''
    A helper function to batch normalize, relu and conv the input layer sequentially
    :param input_layer: 4D tensor
    :param filter_shape: list. [filter_height, filter_width, filter_depth, filter_number]
    :param stride: stride size for conv
    :return: 4D tensor. Y = conv(Relu(batch_normalize(X)))
    '''

   # input_channel = input_layer.get_shape().as_list()[-1]
    #batch_size = input_layer.get_shape().as_list()[0]
    input_size = input_layer.get_shape().as_list()[1]

    #biases = tf.get_variable('biases', [output_channel], initializer=tf.constant_initializer(0.))
    biases = create_variables(name='bias', shape=[output_channel],initializer=tf.constant_initializer(0.))
    relu_layer = lrelu(input_layer)

    if stride ==2:
        # kernel = tf.get_variable('kernel', [filter_size, filter_size, output_channel, input_channel],
        #                          initializer=tf.truncated_normal_initializer(stddev=0.02))
        kernel = create_variables(name='kernel',shape=[filter_size, filter_size, output_channel, input_channel])

        conv_layer = tf.nn.conv2d_transpose(relu_layer, kernel, output_shape=[batch_size, out_size, out_size, output_channel], strides=[1, 2, 2, 1],
                                      padding='SAME')
        # conv_layer = tf.layers.batch_normalization(conv_layer, training=is_training)
    else:
        # kernel = tf.get_variable('kernel', [filter_size, filter_size, input_channel,output_channel],
        #                          initializer=tf.truncated_normal_initializer(stddev=0.02))
        kernel = create_variables(name='kernel', shape=[filter_size, filter_size, input_channel, output_channel])
        conv_layer = tf.nn.conv2d(relu_layer, kernel,strides=[1, 1, 1, 1],padding='SAME')
        # conv_layer = tf.layers.batch_normalization(conv_layer, training=is_training)
    #return conv_layer
    return conv_layer+biases

def inv_residual_block(input_layer, input_channel,output_channel, filter_size,out_size,batch_size,is_training,first_block=False):
    '''
    Defines a residual block in ResNet
    :param input_layer: 4D tensor
    :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
    :param first_block: if this is the first residual block of the whole network
    :return: 4D tensor.
    '''
    #input_channel = input_layer.get_shape().as_list()[-1]
    #batch_size = input_layer.get_shape().as_list()[0]
    input_size = input_layer.get_shape().as_list()[1]

    # When it's time to "shrink" the image size, we use stride = 2
    if input_channel / 2 == output_channel:
        decrease_dim = True
        stride = 2
    elif input_channel == output_channel:
        decrease_dim = False
        stride = 1
    else:
        raise ValueError('Output and input channel does not match in residual blocks!!!')

    # The first conv layer of the first residual block does not need to be normalized and relu-ed.
    with tf.variable_scope('deconv1_in_block'):
        if first_block:
            # kernel = tf.get_variable('kernel', [filter_size, filter_size, output_channel, input_channel], initializer=tf.truncated_normal_initializer(stddev=0.02))
            # biases = tf.get_variable('biases', [output_channel], initializer=tf.constant_initializer(0.))
            kernel = create_variables(name='kernel', shape=[filter_size, filter_size, output_channel, input_channel])
            biases = create_variables(name='bias', shape=[output_channel], initializer=tf.constant_initializer(0.))
            conv1 = tf.nn.conv2d_transpose(input_layer, kernel, output_shape=[batch_size, out_size, out_size, output_channel], strides=[1, 2, 2, 1],
                                          padding='SAME')
            # conv1 = tf.layers.batch_normalization(conv1, training=is_training)
            # conv1 = conv1 + biases
        else:
            conv1 = inv_relu_conv_layer(input_layer,input_channel,output_channel,filter_size,out_size,batch_size,stride,is_training)

    with tf.variable_scope('deconv2_in_block'):
        conv2 = inv_relu_conv_layer(conv1,output_channel,output_channel,filter_size,out_size,batch_size,1,is_training)

    # When the channels of input layer and conv2 does not match, we add zero pads to increase the
    #  depth of input layers
    if decrease_dim is True:
        filter = create_variables(name='filter', shape=[1, 1, input_channel, output_channel])
        padded_input =tf.nn.conv2d(input_layer,filter,[1,1,1,1],padding='SAME')
        #padded_input = tf.slice(input_layer,[0,0,0,0],[-1,-1,-1,input_channel/2])
        padd_size=int(input_size/2)
        padded_input = tf.pad(padded_input, [[0, 0], [padd_size, padd_size], [padd_size, padd_size], [0, 0]])
    else:
        padded_input = input_layer

    output = conv2 + padded_input
    return output



def inference(input_tensor_batch, n, is_training,reuse):
    '''
    The main function that defines the ResNet. total layers = 1 + 2n + 2n + 2n +1 = 6n + 2
    :param input_tensor_batch: 4D tensor
    :param n: num_residual_blocks
    :param reuse: To build train graph, reuse=False. To build validation graph and share weights
    with train graph, resue=True
    :return: last layer in the network. Not softmax-ed
    '''

    layers = []
    image = tf.reshape(input_tensor_batch, [-1, 64, 64, 1])
    # mnist data's shape is (28 , 28 , 1)
    # yb = tf.reshape(cond_info, shape=[128, 1, 1, 22])
    # concat
    # concat_data = conv_cond_concat(image, yb)
    out_channel=32
    with tf.variable_scope('conv0', reuse=reuse):
        # image = tf.reshape(input_tensor_batch,[-1,64,64,1])
        conv0 = conv_bn_relu_layer(image, [5, 5, 1, out_channel], 1, is_training)
        # activation_summary(conv0)
        conv0 = tf.nn.avg_pool(conv0, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


        layers.append(conv0)
    # image=32*32

    for i in range(n):
        with tf.variable_scope('conv1_%d' % i, reuse=reuse):
            if i == 0:
                conv1 = residual_block(layers[-1], out_channel, is_training, first_block=True)
            else:
                conv1 = residual_block(layers[-1], out_channel, is_training)
            # activation_summary(conv1)
            layers.append(conv1)
    # image=32*32
    for i in range(n):
        with tf.variable_scope('conv2_%d' % i, reuse=reuse):
            conv2 = residual_block(layers[-1], out_channel * 2, is_training)
            # activation_summary(conv2)
            layers.append(conv2)
    # image=16*16
    for i in range(n):
        with tf.variable_scope('conv3_%d' % i, reuse=reuse):
            conv3 = residual_block(layers[-1], out_channel * 4, is_training)
            layers.append(conv3)
        # assert conv3.get_shape().as_list()[1:] == [8, 8, 64]
    # image=8*8
    for i in range(n):
        with tf.variable_scope('conv4_%d' % i, reuse=reuse):
            conv4 = residual_block(layers[-1], out_channel * 8, is_training)
            layers.append(conv4)
        #assert conv3.get_shape().as_list()[1:] == [4, 4, 128]
    # image=4*4


    return layers[-1],layers[2]

# Gaussian MLP as encoder
def gaussian_CNN_encoder(x,  dim_z, is_training,keep_prob):


    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    #w_init = tf.contrib.layers.variance_scaling_initializer()
    w_init = tf.truncated_normal_initializer(stddev=0.02)
    b_init = tf.constant_initializer(0.)

    fcl,_= inference(x, 2, is_training,reuse=tf.AUTO_REUSE)

    relu_layer = lrelu(fcl)  # (?,4,4,256)

    fc = tf.reduce_mean(relu_layer, [1, 2])  # (?,256)

    # xc= tf.layers.flatten(fcl)
    # xc1 = tf.layers.dense(xc,512,activation=None)
    out_weight = tf.Variable(tf.truncated_normal([512,2],stddev=0.2))
    # out_bias = tf.Variable(tf.constant(0.1,shape=[2]))
    # svm_output = tf.matmul(xc1,out_weight)+ out_bias
    regularization_loss = tf.reduce_mean(tf.square(out_weight))

    xc = tf.layers.flatten(relu_layer)
    xc1 = tf.layers.dense(xc, 512, activation=tf.nn.relu)
    svm_output = tf.layers.dense(xc1, 2)

    # softmax_linear = tf.layers.dense(xc1, 2)

    with tf.variable_scope('sample', reuse=tf.AUTO_REUSE):

        wo = tf.get_variable('wo', [fc.get_shape()[1], dim_z * 2], initializer=w_init)
        bo = tf.get_variable('bo', [dim_z * 2], initializer=b_init)
        gaussian_params = tf.matmul(fc, wo) + bo

        # The mean parameter is unconstrained
        mean = gaussian_params[:, :dim_z]
        # The standard deviation must be positive. Parametrize with a softplus and
        # add a small epsilon for numerical stability
        stddev = 1e-6 + tf.nn.softplus(gaussian_params[:,dim_z :])


    return mean, stddev, svm_output,xc1, regularization_loss

# Bernoulli MLP as decoder
def gaussian_CNN_decoder(z, batch_size,is_training,dim_z,reuse=False):
 with tf.variable_scope("gaussian_CNN_decoder",reuse=reuse):
    batch_size=tf.cast(batch_size,dtype=tf.int32)           #tf.cast()函数的作用是执行 tensorflow 中张量数据类型转换
    print('b',batch_size)
    # initializers
    #w_init = tf.contrib.layers.variance_scaling_initializer()
    w_init = tf.truncated_normal_initializer(stddev=0.02)       #生成截断正态分布的初始化程序.
    b_init = tf.constant_initializer(0.)


    # # add class info to hidden variable z
    # z = tf.concat([z, cond_info], 1)
    # # 1st hidden layer
    w0 = tf.get_variable('w0', [dim_z + 2, 4 * 4 * 256], initializer=w_init)
    b0 = tf.get_variable('b0', [4 * 4 * 256], initializer=b_init)
    h0 = tf.matmul(z, w0) + b0
    h0 = tf.layers.batch_normalization(h0, training=is_training)
    h0 = lrelu(h0)

    # layers = []
    deconv0 = tf.reshape(h0, [-1, 4, 4, 256])
    # layers.append(deconv0)

    with tf.variable_scope('deconv1_1'):
        deconv1_1 = inv_residual_block(deconv0, 256, 128, 3, 8, batch_size, is_training, first_block=True)
    with tf.variable_scope('deconv1_2'):
        deconv1_2 = inv_residual_block(deconv1_1, 128, 128, 3, 8, batch_size, is_training)
    # image=8*8

    with tf.variable_scope('deconv2_1'):
        deconv2_1 = inv_residual_block(deconv1_2, 128, 64, 3, 16, batch_size, is_training)
    with tf.variable_scope('deconv2_2'):
        deconv2_2 = inv_residual_block(deconv2_1, 64, 64, 3, 16, batch_size, is_training)
    # image=16*16

    with tf.variable_scope('deconv3_1'):
        deconv3_1 = inv_residual_block(deconv2_2, 64, 32, 3, 32, batch_size, is_training)
    with tf.variable_scope('deconv3_2'):
        deconv3_2 = inv_residual_block(deconv3_1, 32, 32, 3, 32, batch_size, is_training)
    # image=32*32

    with tf.variable_scope('deconv4') as scope:
        kernel = tf.get_variable('kernel', [5, 5, 1, 32], initializer=w_init)
        biases = tf.get_variable('biases', [1], initializer=b_init)
        conv = tf.nn.conv2d_transpose(deconv3_2, kernel, output_shape=[batch_size, 64, 64, 1], strides=[1, 2, 2, 1],
                                      padding='SAME')  # 反卷积

        conv4 = tf.nn.tanh(conv + biases)

    y = tf.reshape(conv4, shape=[-1, 64*64])


    return y


# Gateway
def CNN_autoencoder(x,dim_z,label,batch_size,is_training,keep_prob):



    # encoding
    mu, sigma , logit,feature, regular_loss= gaussian_CNN_encoder(x, dim_z, is_training, keep_prob)



    # sampling by re-parameterization technique
    z1 = mu + sigma * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)       #tf.random_normal()函数用于从服从指定正太分布的数值中取出指定个数的值

    z = tf.concat([z1,label],1)


    reconstr = gaussian_CNN_decoder(z,batch_size,is_training,dim_z)

    reconstr_loss_clamped = tf.reduce_sum(tf.square(x-reconstr),1)
    reconstr_loss_clamped = tf.reduce_mean(reconstr_loss_clamped)



    KL_divergence = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(sigma) - tf.log(1e-8 + tf.square(sigma)) - 1, 1)
    KL_divergence = tf.reduce_mean(KL_divergence)



    cost_clamped = ( reconstr_loss_clamped +0.1 * KL_divergence )

    return z,  KL_divergence, mu, sigma,logit,feature, regular_loss, cost_clamped,reconstr_loss_clamped,reconstr

