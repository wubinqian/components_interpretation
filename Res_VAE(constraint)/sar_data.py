
from __future__ import absolute_import
from __future__ import division
from __future__ import  print_function

import os
import copy
import numpy as np
import scipy.io as sio
import tensorflow as tf

# param for sar
IMAGE_SIZE=64
NUM_CHANNELs = 1
PIXEL_DEPTH = 1.0
NUM_LABELS = 2
VALIDATION_SIZE = 5000



def dense_to_one_hot(label_dense,num_classes=2):
    """Convert class labels from scalars  to one-hot vectors"""         # scalars 标量  一个one-hot向量除了某一位的数字是1以外其余各维度数字都是0
    num_labels = label_dense.shape[0]
    index_offset = np.arange(num_labels)*num_classes          # idnex_offset该下标表表示的是一维时候每个labels的对应下标  arange一个参数时，参数值为终点，起点取默认值0，步长取默认值1
    labels_one_hot = np.zeros((num_labels,num_classes))
    labels_one_hot.flat[index_offset+label_dense.ravel()] = 1   # 对one_hot矩阵的指定的位置进行赋值1的操作  index_offset+labels_dense.ravel() 得到的是一个下标   flat属性返回的是一个array的遍历对象，此时它是一维形式的  ravel()返回的是一个副本，但是这个副本是原来数据的引用，有点类似于c++的引用。主要是减少存储空间的使用。返回的也是一个一维形式的数据
    return labels_one_hot

# def expend_total_data(images, labels,tag=True):
#     expanded_images = []
#     expanded_labels = []
#
#     j = 0  # counter
#     for x, y in zip(images, labels):
#         j = j + 1
#         if j % 100 == 0:
#             print('expanding data : %03d / %03d' % (j, numpy.size(images, 0)))
#
#         image = numpy.reshape(x, (96, 96))
#         cent_image = image[16:80, 16:80]
#
#         # register original data
#         expanded_images.append(numpy.reshape(cent_image,64**2))          #list.append(object),list 	待添加元素的列表   object 	将要给列表中添加的对象 	不可省略的参数
#         expanded_labels.append(y)
#
#         # get a value for the background
#         # zero is the expected value, but median() is used to estimate background's value
#         bg_value = numpy.median(x)  # this is regarded as background's value
#         # image = numpy.reshape(x, (IMAGE_SIZE, IMAGE_SIZE))
#
#         if tag:
#          for i in range(2):
#             # rotate the image with random degree
#             angle = numpy.random.randint(-15, 15, 1)
#             new_img_ = ndimage.rotate(image, angle, reshape=False, cval=bg_value)
#             new_img_ = new_img_[16:80, 16:80]
#             # shift the image with random distance
#             #down_shift = numpy.random.randint(12, 22, 1)[0]
#             #right_shift = numpy.random.randint(10, 24, 1)
#             #new_img_ = image[down_shift:down_shift+64,down_shift:down_shift+64]
#
#             # register new training data
#             expanded_images.append(numpy.reshape(new_img_, 64 ** 2))
#             expanded_labels.append(y)
#
#     # images and labels are concatenated for random-shuffle at each epoch
#     # notice that pair of image and label should not be broken
#     expanded_train_total_data = numpy.concatenate((expanded_images, expanded_labels), axis=1)
#     numpy.random.shuffle(expanded_train_total_data)
#
#     return expanded_train_total_data


def extractdb_images(filename,tag):

    imdb = sio.loadmat(filename)
    if tag==1:
        data = imdb.get('trainData')
    elif tag==2:
        data = imdb.get('valData')
    elif tag==3:
        data = imdb.get('testData')
    elif tag==4:
        data = imdb.get('trainimdb')
    else:
        data = imdb.get('newimdb')

    data = np.asarray(data)
    # print(data.shape[0])         #2304           #576
    data = data.reshape([data.shape[0],64,64,1])
    # print(data.shape)         #(2304,64,64,1)    #(576,64,64,1)
    return data

def extractdb_labels(filename,tag,one_hot):

    imdb = sio.loadmat(filename)
    if tag == 1:
        labels = imdb.get('trainLabel')
    elif tag == 2:
        labels = imdb.get('valLabel')
    else:
        labels = imdb.get('testLabel')
    labels = np.asarray(labels)
    # labels = labels-1
    # print(labels)     # 11111.....000000
    if one_hot:
        return tf.one_hot(labels[:, 0], depth=2, axis=1)
    return labels






def extract_data(filename,norm_shift=False,norm_scale=True,tag=1):
    """extract the images into 4D tensor [image_ndex,y,x,channels].
    Values are rescaled from [0,255] down to [-0.5,0.5]"""
    print('Extracting',filename)
    data = extractdb_images(filename,tag)

    if norm_shift:
        data = data-(PIXEL_DEPTH/2.0)
    if norm_scale:
        data = data/PIXEL_DEPTH

    num = data.shape[0]
    data = np.reshape(data,[num,-1])
    # print(data.shape)          #(2304,4096)          #(576,4096)

    return data


def extract_labels(filename,tag,one_hot):
    """Extract labels into a vector of int64 label IDs"""
    print('Extracting labels',filename)
    return extractdb_labels(filename,tag,one_hot=one_hot)



def prepare_sar_data(use_norm_shift=False,use_norm_scale=True,use_data_augumentation=False):    # data_augumentation   数据增强

    total_data = extract_data("./data/trainData",use_norm_shift,use_norm_scale,1)
    total_data_len = total_data.shape[0]                         #  2304  shape[0]就是读取矩阵第一维度的长度,相当于行数
    total_labels = extract_labels("./data/trainLabel",tag=1,one_hot=True)

    total_data = np.reshape(total_data,[total_data_len,-1])      #-1表示自动计算新数组的列数   (2304,4096)
    # print(total_data.shape)

    test_data = extract_data("./test_data/trainData", use_norm_shift, use_norm_scale, 1)     #(576,4096)
    test_data_len = test_data.shape[0]
    test_labels = extract_labels("./test_data/trainLabel", tag=1, one_hot= True)   # (576,1)  11111...1111


    test_data = np.reshape(test_data, [test_data_len, -1])  #(576,4096)

    total_labels2 = extract_labels("./data/trainLabel", tag=1, one_hot=False)
    test_labels2 = extract_labels("./test_data/trainLabel", tag=1, one_hot=False)  # (576,1)  11111...1111

    # print('train data size:  ' ,(total_data.shape))
    # print('train label :  ', total_labels)
    #
    # print('train label size:  ', (total_labels.shape))
    # print('train label type:  ', (total_labels.dtype))
    #
    # print('test data size: ' ,(test_data.shape))
    # print('test labels size: ' ,(test_labels.shape))
    # print('test data size: ' , (test_data.shape))
    # print('test labels size:' , (test_labels.shape))


    return total_data, total_labels,test_data,test_labels,total_labels2, test_labels2


def prepare_sar_data2(use_norm_shift=False,use_norm_scale=True,use_data_augumentation=False):

    total_data = extract_data("./data/trainData",use_norm_shift,use_norm_scale,1)
    total_data_len = total_data.shape[0]
    total_labels = extract_labels("./data/trainLabel",tag=1)

    total_data = np.reshape(total_data,[total_data_len,-1])






    test_data = extract_data("./test_data/trainData", use_norm_shift, use_norm_scale, 1)
    test_data_len = test_data.shape[0]
    test_labels = extract_labels("./test_data/trainLabel", tag=1)

    test_data = np.reshape(test_data, [test_data_len, -1])


    # index = np.arange(0,128)
    # tmp = total_data[index,:]
    # test_data = copy.deepcopy(tmp)
    # tmp = total_labels[index,:]
    # test_labels = copy.deepcopy(tmp)

    # if use_data_augumentation:
    #     train_total_data = expend_total_data(total_data,total_labels,False)
    # else:
    #     train_total_data = np.concatenate((total_data,total_labels),axis=1)
    #
    # train_size = train_total_data.shape[0]
    # print('training data size:%03d' %(train_size))
    # return train_total_data,train_size,test_data,test_labels
    print('training data size: %03d ' % (total_data.shape[0]))
    print('test data size:%03d' %(test_data.shape[0]))
    return total_data, total_labels, test_data, test_labels




def prepare_MNIST_data(use_norm_shift=False, use_norm_scale=True):
    # Get the data.

    total_data = extract_data("./data/trainData", use_norm_shift, use_norm_scale, 1)
    total_data_len=total_data.shape[0]
    total_labels = extract_labels("./data/trainLabel", tag=1)

    # AangleDb = sio.loadmat("./data/trainAangle")
    # trainAangle = AangleDb.get('trainAangle')
    # sin_Aangle = numpy.sin(trainAangle/180.0*numpy.pi)
    # cos_Aangle = numpy.cos(trainAangle/180.0*numpy.pi)

    total_data= np.reshape(total_data, [total_data_len, -1])

    index=np.arange(0,128)
    tmp=total_data[index,:]
    test_data=copy.deepcopy(tmp)
    tmp = total_labels[index, :]
    test_labels = copy.deepcopy(tmp)
    # tmpsin =sin_Aangle[index, :]
    # tmpsin = copy.deepcopy(tmpsin)
    # tmpcos = cos_Aangle[index, :]
    # tmpcos = copy.deepcopy(tmpcos)

    # test_labels = numpy.concatenate((test_labels, tmpsin), axis=1)
    # test_labels = numpy.concatenate((test_labels, tmpcos), axis=1)
    #
    # total_labels = numpy.concatenate((total_labels, sin_Aangle), axis=1)
    # total_labels = numpy.concatenate((total_labels, cos_Aangle), axis=1)

    print('training data size: %03d ' % (total_data.shape[0]))
    return total_data, total_labels, test_data, test_labels


def read_data_sets(fake_data=False, one_hot=False, dtype=tf.float32):
  class DataSets(object):
    pass
  data_sets = DataSets()
  if fake_data:
    def fake():
      return DataSet([], [], fake_data=True, one_hot=one_hot, dtype=dtype)
    data_sets.train = fake()
    data_sets.validation = fake()
    data_sets.test = fake()
    return data_sets

  train_data, train_labels, test_data, test_labels=prepare_sar_data2(use_norm_shift=False, use_norm_scale=True)
  data_sets.train = DataSet(train_data, train_labels, dtype=dtype)
  data_sets.test = DataSet(test_data, test_labels, dtype=dtype)
  return data_sets

class DataSet(object):
  def __init__(self, images, labels, fake_data=False, one_hot=False,
               dtype=tf.float32):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    dtype = tf.as_dtype(dtype).base_dtype          #返回该DType的基础DType，而非参考的数据类型(non-reference)
    if dtype not in (tf.uint8, tf.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
    else:
      assert images.shape[0] == labels.shape[0], (
          'images.shape: %s labels.shape: %s' % (images.shape,
                                                 labels.shape))
      self._num_examples = images.shape[0]
      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)

      # assert images.shape[3] == 1
      # images = images.reshape(images.shape[0],
      #                         images.shape[1] * images.shape[2])

    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0
  @property              #@property装饰器就是负责把一个方法变成属性调用的：
  def images(self):
    return self._images
  @property
  def labels(self):
    return self._labels
  @property
  def num_examples(self):
    return self._num_examples
  @property
  def epochs_completed(self):
    return self._epochs_completed
  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]


if __name__ == '__main__':


    train_data, train_label,test_data2,test_labels2,test_data, test_labels = prepare_sar_data()
    train_label=train_label.flatten()
    print(train_data.shape,train_label.shape)
    print(test_data.shape,test_labels.shape)

    print(np.max(train_data),np.min(train_data),np.mean(train_data))
    print(np.max(test_data),np.min(test_data),np.mean(test_data))
