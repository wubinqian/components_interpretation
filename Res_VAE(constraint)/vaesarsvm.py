# coding:utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sar_data
import os
import cnnvae
import cnnvaelv
import model
import plot_utils
import scipy.io as sio         #读取.mat文件
from sklearn import  svm
from sklearn.preprocessing import MinMaxScaler


"""hyper parameter"""
batch_size=64#minibatch,original sets 20
# n_z = 2 #Dimension of the latent space
dim_z=16
# learn_rate = 0.0005
n_epochs =120
controlled_z = 1
penalty_parameter =1
PRR = False
ADD_NOISE = False
PMLR = False


"""prepare sar data"""

# train_data, train_label,test_data2,test_labels2 = mnist_data.prepare_MNIST_data()
train_data, train_label, test_data, test_labels, train_labels2, test_labels2 = sar_data.prepare_sar_data()
# test_data = test_data[22:43]
# test_labels = test_labels[22:43]
# test_labels2 = test_labels2[22:43]


train_data2 = train_data
n_samples = train_data.shape[0]
test_samples = test_data.shape[0]



"""build graph"""
# input placeholder
train_x = tf.placeholder(tf.float32, shape=[None, 64 * 64], name='target_img')

# keep prob
keep_prob = tf.placeholder(tf.float32,shape=[],name='keep_prob')
is_training = tf.placeholder(tf.bool)                 # tf.bool:布尔型bool取值false和true

label = tf.placeholder(tf.float32,shape=[None,2],name='label') # transfer to one-hot
# test_label = tf.placeholder(tf.float32, shape=[None, 2], name='test_label')
# labelz = tf.placeholder(tf.float32,shape=[None , 1],name='labelz')
# y_ = tf.placeholder(tf.float32, shape=[None, 10])   #Batchsize x 10 (one hot encoded)

kl_epoch = tf.placeholder(tf.float32, name='kl_epoch')

# input for PMLR
z_in = tf.placeholder(tf.float32, shape=[None, dim_z], name='latent_variable')

# network architecture
z, KL_divergence, z_mean, z_sigma, x_hat,x_feature, regularization_loss, cost_clamped, neg_reconstr_loss_clamped, y_clamped= cnnvae.CNN_autoencoder(train_x, dim_z, label, batch_size=batch_size, is_training=is_training, keep_prob= keep_prob)
# print('4', y_hat.shape)
# _, _,y_hat = cnnvae.gaussian_CNN_encoder(test_y,  dim_z, is_training=False,keep_prob=keep_prob)
prediction = tf.equal(tf.argmax(x_hat, 1), tf.argmax(label, 1))
acc = tf.reduce_mean(tf.cast(prediction,tf.float32))

corret_prediction = tf.argmax(x_hat, 1)
pre_label = tf.cast(corret_prediction,tf.float32)
# lossc = tf.nn.softmax_cross_entropy_with_logits_v2(logits=x_hat, labels=label)

lossc = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_hat, labels=label)
loss_cross = tf.reduce_mean(lossc)


cost_clamp_summary = tf.summary.scalar('loss', loss_cross+cost_clamped)     #用来显示标量信息
# tf.summary.histogram('histogram loss', cost_clamped)              #显示直方图信息
# summary_op = tf.summary.merge_all()                            #用这一句就可一显示训练时的各种信息了

global_ = tf.Variable(tf.constant(0))
learn_rate = tf.train.exponential_decay(0.0005, global_, 100, 0.97, staircase=True)          #指数衰减下降

extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)   #tf.GraphKeys.UPDATE_OPS，这是一个tensorflow的计算图中内置的一个集合，其中会保存一些需要在训练操作之前完成的操作，并配合tf.control_dependencies函数使用
with tf.control_dependencies(extra_update_ops):                  #tf.control_dependencies，该函数保证其辖域中的操作必须要在该函数所传递的参数中的操作完成后再进行

    optimizer_clamped = tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(loss_cross+cost_clamped)


"""training"""

loss_array = np.zeros(shape=[n_epochs, 1], dtype=np.float32)
epoch_array = np.zeros(shape=[n_epochs, 1], dtype=np.uint)

total_batch = int(n_samples/batch_size)    #2304/128=18
test_batch = int(test_samples / batch_size )
min_total_loss = 1e99
min_total_marginal_loss = 1e99


T_C = []
T_S = []

# tf.reset_default_graph()  # 重置默认图
# graph = tf.Graph()        # 新建空白图
# with graph.as_default() as g:

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())
    #
    train_label = train_label.eval()
    test_labels = test_labels.eval()
    # to visualize using TensorBoard
    # writer = tf.summary.FileWriter('./graphs', sess.graph)

    log_dir = './logs'
    journalist = tf.summary.FileWriter(
        log_dir,
        flush_secs=10             #每多少秒像disk中写数据,并清空对象缓存
    )
    print("Saving tensorboard logs to %r" % (log_dir,))


    for epoch in range(n_epochs):
        # sess.graph.finalize()
        total_loss_likelihood = 0.0
        total_loss_divergence = 0.0
        total_loss_mse = 0.0
        total_loss_cross = 0.0
        # total_tra_acc = 0.0
        test_acc = 0.0

        # index = [j for j in range(train_data.shape[0])]
        r = np.random.permutation(n_samples)
        train_data = train_data[r]
        # print(r.shape)
        # print('oooo',train_data.shape)
        # print(train_label.shape)

        train_label = train_label[r]
        train_labels2 = train_labels2[r]


        test_feed = {train_x : test_data, label: test_labels, keep_prob: 1, is_training: False}

        # Loop over all batches
        for i in range(total_batch):

            batch_xs_input = train_data[(i * batch_size) :((i+1)* batch_size), :]
            # batch_labelz = train_label[(i * batch_size) :((i+1)* batch_size), :]
            # batch_label =sess.run( tf.squeeze(batch_labelz))
            batch_label = train_label[(i * batch_size) :((i+1)* batch_size),:]
            # batch_label2 = train_labels2[(i * batch_size):((i + 1) * batch_size), :]
            # print('1', batch_xs_input.shape)
            # print('2', batch_label.shape)



            _, tot_loss, loss_likelihood, loss_divergence, loss_crossentropy, cost_summary, recon= sess.run(
                (optimizer_clamped, cost_clamped, neg_reconstr_loss_clamped, KL_divergence,loss_cross, cost_clamp_summary, y_clamped),
                feed_dict={train_x: batch_xs_input, label:batch_label, kl_epoch:epoch, is_training: True, global_:epoch, keep_prob:1})

            # x_temp = []
            # for g in batch_xs_input:
            #     x_temp.append(sess.run(x_feature, feed_dict={train_x: np.array(g).reshape((1, 4096))})[0])
            #
            # clf = svm.SVC(C=0.9, kernel='rbf', gamma='auto')
            # clf.fit(x_temp, batch_label2.ravel())



            T_c = sess.run(learn_rate, feed_dict={global_: epoch})
            T_C.append(T_c)

            T_s = sess.run(loss_cross, feed_dict={train_x: batch_xs_input, label:batch_label, kl_epoch:epoch, is_training: True, global_:epoch, keep_prob : 1})
            T_S.append(T_s)


            total_loss_likelihood = total_loss_likelihood + loss_likelihood
            total_loss_divergence = total_loss_divergence + loss_divergence
            total_loss_cross = total_loss_cross + loss_crossentropy
            # total_tra_acc = total_tra_acc + tra_acc

        journalist.add_summary(cost_summary, epoch)
        journalist.flush()


        total_loss_likelihood = total_loss_likelihood / total_batch
        total_loss_divergence = total_loss_divergence / total_batch
        total_loss_cross = total_loss_cross / total_batch
        # total_tra_acc = total_tra_acc / total_batch


        tot_loss =  total_loss_likelihood + total_loss_cross + total_loss_divergence

        epoch_array[epoch] = epoch
        loss_array[epoch] = total_loss_likelihood

        # print cost every epoch
        print("epoch %d: total_loss_cross %03.2f  " % (
            epoch, total_loss_cross))
    print("save model...")
    saver = tf.train.Saver()
    saver.save(sess, './checkpoint/model.ckpt')

###################################  SOFTMAX  #################################

    # saver = tf.train.import_meta_graph('./checkpoint/model.ckpt.meta')
    # # graph = tf.get_default_graph()
    # # 获取预测操作
    # # acc = graph.get_tensor_by_name("acc:0")
    # # 获取输入占位符
    # # train_x = graph.get_tensor_by_name("target_img:0")
    # saver.restore(sess, tf.train.latest_checkpoint('./checkpoint'))
    # print('restored')

    test_feed = {train_x: test_data, label: test_labels, keep_prob: 1, is_training: False}
    pre_y, test_acc= sess.run((pre_label,acc), feed_dict=test_feed)
    print('softmax test accuracy', test_acc)

    ###########################################################3################

############################## SVM  #############################################
    x_temp = []
    for g in train_data :
        x_temp.append(sess.run(x_feature, feed_dict = {is_training : False, train_x: np.array(g).reshape((1, 4096))})[0])

    x_temp2 = []
    for g in test_data:
        x_temp2.append(sess.run(x_feature,feed_dict = {is_training:False, train_x: np.array(g).reshape((1, 4096))})[0] )


    clf = svm.SVC(C=0.9, kernel='rbf',gamma = 'auto')
    clf.fit(x_temp, train_labels2.ravel())
    print ('svm testing accuracy:')
    print (clf.score(x_temp2, test_labels2))
##########################################################################################

#####################################  KNN ########################################
    x_temp3 = sess.run(x_feature, feed_dict= {train_x : train_data, is_training :False})
    x_temp4 = sess.run(x_feature, feed_dict={train_x: test_data, is_training: False})
    # print("训练集特征维度为：",x_temp.shape)
    # print("测试集特征维度为：",x_temp2.shape)
    minMax = MinMaxScaler()
    # 将数据进行归一化
    tra_fea = minMax.fit_transform(x_temp3)
    te_fea = minMax.fit_transform(x_temp4)
    print("训练集归一化后特征维度为：", tra_fea.shape)
    print("测试集归一化后特征维度为：", te_fea.shape)
    xtr = tf.placeholder("float", [None, 512])
    xte = tf.placeholder("float", [512])
    distance = tf.reduce_sum(tf.square(xtr - xte), reduction_indices=1)  # 欧氏距离准确率更高 95%  样本特征要做归一化处理
    # 获取最小距离的索引
    pred = tf.arg_min(distance, 0)

    # 分类精确度
    accuracy = 0.

    # 初始化变量
    init = tf.global_variables_initializer()

    # 运行会话，训练模型
    with tf.Session() as sess:

        # 运行初始化
        sess.run(init)

        # 遍历测试数据
        for i in range(len(te_fea)):
            # 获取当前样本的最近邻索引
            nn_index = sess.run(pred, feed_dict={xtr:tra_fea, xte: te_fea[i,:]})  # 向占位符传入训练数据

            # 计算精确度
            if np.argmax(train_label[nn_index]) == np.argmax(test_labels[i]):
                accuracy += 1. / len(te_fea)

        print("Done!")
        print("KNN Accuracy:", accuracy)
###################################################################################################


    # fig2 = plt.figure()
    # ax2 = fig2.add_subplot(111)
    # ax2.plot(T_C, 'k-')
    # ax2.set_xlabel('step')
    # ax2.set_ylabel('Learning_rate')
    # fig2.suptitle('Learning_rate')
    # plt.savefig('./learn_rate_result.png')
    #
    # plt.show()
    #
    # fig3 = plt.figure()
    # ax3 = fig3.add_subplot(111)
    # ax3.plot(T_S, 'k-')
    # ax3.set_xlabel('step')
    # ax3.set_ylabel('loss')
    # fig3.suptitle('y_cross_loss')
    # plt.savefig('./clamp_loss_result.png')
    #
    # plt.show()
