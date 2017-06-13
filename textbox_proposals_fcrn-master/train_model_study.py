#coding=utf-8
from __future__ import print_function

import itertools
import h5py
import numpy as np
import os
import random
import sys

from random import shuffle

#Keras提供了两种后端引擎Theano/Tensorflow，并将其函数统一封装，使得用户可以以同一个接口调用不同后端引擎的函数
from keras import backend as K
import theano.tensor as T
import theano
import keras.models
import keras.callbacks

from theano.compile.sharedvalue import shared
from keras.models import Sequential, Model
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Activation, Flatten, Input, ZeroPadding2D
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.optimizers import SGD
from keras.utils import np_utils

img_rows = 512
img_cols = 512
nb_epoch = 1000
iteration_size = 100000
mini_batch_size = 12 #输入时，一次输入一批图像的大小
delta = 16
initial_discount = 0.01
discount_step = 0.1

num_samples_per_epoch = 50000 #训练集的数目
num_validation_samples = 5000 #验证集的数目

d = shared(initial_discount, name = 'd')#共享变量，就是各线程，公共拥有的变量，通过get_value()、set_value()可以查看、设置共享变量的数值。 http://blog.csdn.net/hjimce/article/details/46806923

def fcrn_loss(y_true, y_pred):
  loss = K.square(y_pred - y_true) #keras backend square(逐元素平方)  7*16*16
  
  images = []
  
  #如果某一个栅格 predictor 中没有 ground-truth 文本，这个 loss会忽略掉除了 c（text/non-text） 以外的所有 params。 
  for i in range(0, mini_batch_size): #计算一批图像的均值
    c_true = y_true[i, 6, :,:].reshape((1, delta, delta)) #confidence, 1*16*16  # The last feature map in the true vals is the 'c' matrix

    #T.set_subtensor():将d.get_value()==initial_discount的值赋给c_true中<=0.0
    c_discounted = T.set_subtensor(c_true[(c_true<=0.0).nonzero()], d.get_value()) #(c_true<=0.0).nonzero():取c_true<=0.0的下标，c_true[(c_true<=0.0).nonzero()]：取c_true<=0.0的值
    
    final_c = (c_discounted * loss[i,6,:,:])
       
    # Element-wise multiply of the c feature map against all feature maps in the loss
    #(x-u,y-v,w,h,cos,sin) = c_true * loss
    final_loss_parts = [(c_true * loss[i, j, :, :].reshape((1, delta, delta))).reshape((1, delta, delta)) for j in range(0, 6)]
    final_loss_parts.append(final_c) #final_loss_parts是含有7个数的列表
    
    images.append(K.concatenate(final_loss_parts)) #concatenate 连接，concatenate(tensors, axis=-1)在给定轴上将一个列表中的张量串联为一个张量 specified axis
    tt = K.mean(K.concatenate(images).reshape((mini_batch_size, 7, delta, delta)), axis = 1)
    print (tt) #mean
  return K.mean(K.concatenate(images).reshape((mini_batch_size, 7, delta, delta)), axis = 1) #12*7*16*16， mean(x, axis=None, keepdims=False)：在给定轴上求张量元素之均值 http://keras-cn.readthedocs.io/en/latest/backend/?highlight=mean
  # labels =  K.zeros((12,7, 16, 16)) la = K.mean(labels, axis = 1) mm=(la.eval()) print mm.shape 12*16*16
  #if 大小(1,2,3,4) axis=0->(2,3,4) axis=1->(1,3,4) axis=-1->(1,2,3) 是按照列表的访问方式的，如果省略axis参数，就是个数字
  
class DiscountCallback(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    print("Running callback: " + str(epoch))
    d.set_value(d.get_value() + discount_step)#shared(initial_discount, name = 'd')
    
def build_model():
  if os.path.exists(model_file + ".h5"): #如果path存在，返回True；如果path不存在，返回False。
    print("Loading saved model for incremental training...")
    model = keras.models.load_model(model_file + ".h5", custom_objects = {'fcrn_loss', fcrn_loss}) #custom_objects自定义对象，fcrn_loss目标函数，load_model()参数还没弄清楚？？？
    #使用keras.models.load_model(filepath)来重新实例化你的模型，如果文件中存储了训练配置的话，该函数还会同时完成模型的编译
  else:
    model = Sequential() #初始化一个神经网络，（Sequential是顺序模型，多个网络层的线性堆叠）
    
    # Layer 1 #http://m.blog.csdn.net/article/details?id=52678453 网络层的讲解可参考，本代码基于theano写的
    model.add(ZeroPadding2D(padding = (2, 2), input_shape=(1, img_rows, img_cols))) #theano (channels,w,h) ,tensorflow (w,h,channels)#在卷积前使用ZeroPadding2D进行补零操作，padding = (rows, cols)rows表示会在行的最前和最后都增加rows行0,img_width += 2*rows
    model.add(Convolution2D(64, 5, 5))     #(nb_filter,nb_row,nb_col) num_filters: 64, kernel size: 5×5, stride: 1, dim_ordering：'th'或'tf'，th=theano,tf=tensorflow
    model.add(BatchNormalization(axis = 1))#axis=1，对每个特征图进行规范化，#该层在每个batch上将前一层的激活值重新规范化，即使得其输出数据的均值接近0，其标准差接近1。
    model.add(Activation('relu'))          #激活层，relu = max(x,0)
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2,2)))#迟化层，压缩特征维度，把相邻的区域变成一个值，方法：最大化、平均化、随机，pool_size(rows,cols)rows、cols为2的整数倍，代表在竖直、垂直方向上的下采样，图片大小变化：img_width/2，img_height/2, stride=2(一般为2)，
    print("Layer 1: " + str(model.layers[-1].output_shape))#输出数据的形状
    
    # Layer 2
    model.add(ZeroPadding2D(padding = (2, 2)))
    model.add(Convolution2D(128, 5, 5))
    model.add(BatchNormalization(axis = 1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2,2)))
    print("Layer 2: " + str(model.layers[-1].output_shape))
    
    # Layer 3
    model.add(ZeroPadding2D(padding = (1, 1)))
    model.add(Convolution2D(128, 3, 3))
    model.add(BatchNormalization(axis = 1))
    model.add(Activation('relu'))
    print("Layer 3: " + str(model.layers[-1].output_shape))
 
    # Layer 4
    model.add(ZeroPadding2D(padding = (1, 1)))
    model.add(Convolution2D(128, 3, 3))
    model.add(BatchNormalization(axis = 1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2,2)))
    print("Layer 4: " + str(model.layers[-1].output_shape))
    
    # Layer 5
    model.add(ZeroPadding2D(padding = (1, 1)))
    model.add(Convolution2D(256, 3, 3))
    model.add(BatchNormalization(axis = 1))
    model.add(Activation('relu'))
    print("Layer 5: " + str(model.layers[-1].output_shape))
    
    
    # Layer 6
    model.add(ZeroPadding2D(padding = (1, 1)))
    model.add(Convolution2D(256, 3, 3))
    model.add(BatchNormalization(axis = 1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2,2)))
    print("Layer 6: " + str(model.layers[-1].output_shape))
    
    
    # Layer 7
    model.add(ZeroPadding2D(padding = (1, 1)))
    model.add(Convolution2D(512, 3, 3))
    model.add(BatchNormalization(axis = 1))
    model.add(Activation('relu'))
    print("Layer 7: " + str(model.layers[-1].output_shape))
    
    # Layer 8
    model.add(ZeroPadding2D(padding = (1, 1)))
    model.add(Convolution2D(512, 3, 3))
    model.add(BatchNormalization(axis = 1))
    model.add(Activation('relu'))
    print("Layer 4: " + str(model.layers[-1].output_shape))
    
    
    # Layer 9
    model.add(ZeroPadding2D(padding = (2, 2)))
    model.add(Convolution2D(512, 5, 5))
    model.add(BatchNormalization(axis = 1))
    model.add(Activation('relu'))
    print("Layer 9: " + str(model.layers[-1].output_shape)) 
    #layer1~9： 得到输入图像的feature map （512*32*32）

    # Layer 10 #输出层
    model.add(ZeroPadding2D(padding = (2, 2)))
    model.add(Convolution2D(7, 5, 5))#7个参数：(x-u,y-v,w,h,cos,sin，confidence)
    model.add(BatchNormalization(axis = 1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2,2)))
    print("Layer 10: " + str(model.layers[-1].output_shape)) #7*16*16
    
    #lr表示学习速率,momentum表示动量项，decay是学习速率的衰减系数(每个epoch衰减一次),Nesterov的值是False或者True，表示使不使用Nesterov momentum。http://keras-cn.readthedocs.io/en/latest/other/optimizers/?highlight=compile
    sgd = SGD(lr = 10e-4, decay = 5e-4, momentum = 0.9, nesterov = False)#SGD随机梯度下降，优化算法 
    
    model.compile(loss = fcrn_loss, optimizer = sgd, metrics=['accuracy']) #编译，指定目标函数（loss）与优化方法， metrics：列表，包含评估模型在训练和测试时的性能的指标，典型用法是metrics=['accuracy']http://keras-cn.readthedocs.io/en/latest/models/model/?highlight=model.compile
    
  return model
        
def batch(iterable, n = 1): #加载一批图像（mini_batch_size=12）
  current_batch = []
  for item in iterable:
    current_batch.append(item)
    if len(current_batch) == n:
      yield current_batch
      current_batch = []
      
def exemplar_generator(db_iters, batch_size): #batch_size=mini_batch_size=12
  while True:
    for chunk in batch(itertools.chain.from_iterable(db_iters), batch_size):
      X = []
      Y = []
      
      for item in chunk:
        X.append(item[:].reshape(1, img_rows, img_cols))
        labels = np.array(item.attrs['label']).transpose(2, 0, 1)
        Y.append(labels.reshape(7, delta, delta))  
        
      yield (np.array(X), np.array(Y))
      
def load_db(db_filename):
  try:
    db = h5py.File(db_filename, 'r')
    return db['data'].itervalues() #db['data']是h5py._hl.group.Group,是个生成器
  except:
    print(sys.exc_info()[1]) #sys.exc_info()：获取当前正在处理的异常类
    return []
  
def load_exemplars(db_path):
  print "load_exemplars ........."  #lambda x: db_path + "/" + x  lambda x：匿名函数，函数y=db_path + "/" + x  https://www.zhihu.com/question/20125256
  dbs = map(lambda x: db_path + "/" + x, [f for f in os.listdir(db_path) if os.path.isfile(db_path + "/" + f)]) #isfile(是否是存在的文件)listdir(返回指定路径下文件和文件夹的列表
  return exemplar_generator(map(lambda x: load_db(x), dbs), mini_batch_size) #mini_batch_size=12

if __name__ == '__main__':
  model_file = "bb-fcrn-model"
  train_db_path = "./data" #"/path/to/dbs"
  validate_db_path = "./data" #"/path/to/dbs"
  
  print("Loading data...")
  
  train = load_exemplars(train_db_path) #训练集, 生成器（一次加载一批数据，不需要将所有数据都加载进去）
  validate = load_exemplars(validate_db_path) #验证集，生成器
  
  print("Data loaded.")
  print("Building model...")
  
  model = build_model()
  
  #该回调函数将在每个epoch后保存模型到model_file http://keras-cn.readthedocs.io/en/latest/other/callbacks/
  #save_weights_only：若设置为True，则只保存模型权重，否则将保存整个模型（包括模型结构，配置信息等）
  #mode：‘auto’，‘min’，‘max’之一，在save_best_only=True时决定性能最佳模型的评判准则，例如，当监测值为val_acc时，模式应为max，当检测值为val_loss时，模式应为min。在auto模式下，评价准则由被监测值的名字自动推断。
  checkpoint = keras.callbacks.ModelCheckpoint(model_file + ".h5",
                                               monitor = "acc",
                                               verbose = 1,
                                               save_best_only = True,
                                               save_weights_only = False,
                                               mode = 'auto')
                                               
  #EarlyStopping：早期停止，当监测值不再改善时，该回调函数将中止训练，monitor：需要监视的量，verbose：信息展示模式
  #patience：当early stop被激活（如发现loss相比上一个epoch训练没有下降），则经过patience个epoch后停止训练。
  #mode：‘auto’，‘min’，‘max’之一，在min模式下，如果检测值停止下降则中止训练。在max模式下，当检测值不再上升则停止训练。
  earlystopping = keras.callbacks.EarlyStopping(monitor = 'loss',
                                                min_delta = 0,
                                                patience = 5,
                                                verbose = 1,
                                                mode = 'auto')
  
  discount = DiscountCallback() #？？？
  
  csvlogger = keras.callbacks.CSVLogger(model_file + "-log.csv", append = True) #将epoch的训练结果保存在csv文件中，支持所有可被转换为string的值，包括1D的可迭代数值如np.ndarray.http://keras-cn.readthedocs.io/en/latest/other/callbacks/
  
  #model.fit_generator：利用Python的生成器，逐个生成数据的batch并进行训练。生成器与模型将并行执行以提高效率。http://keras-cn.readthedocs.io/en/latest/models/model/?highlight=model.compile
  model.fit_generator(train,
                      samples_per_epoch = num_samples_per_epoch,
                      nb_epoch = nb_epoch,
                      verbose = 1,
                      validation_data = validate,
                      nb_val_samples = num_validation_samples,
                      max_q_size = 10,
                      pickle_safe = True,
                      callbacks = [checkpoint, earlystopping, csvlogger, discount])
                      
     #train:是一个generator：生成器函数,所有的返回值都应该包含相同数目的样本。生成器将无限在数据集上循环。每个epoch以经过模型的样本数达到samples_per_epoch时，记一个epoch结束
     #nb_epoch:epoch时期，样本循环次数，在一个epoch结束之后，下一个epoch开始之前，将所有的训练集打乱，然后生成新的batch，继续循环
     #samples_per_epoch: 50000, 整数，当模型处理的样本达到此数目时计一个epoch结束，执行下一个epoch     
     #verbos:日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
     #validation_data:生成验证集的生成器， 验证集  
     #nb_val_samples：验证集数目，仅当validation_data是生成器时使用，用以限制在每个epoch结束时用来验证模型的验证集样本数
     #max_q_size ：生成器队列的最大容量     
     #callbacks：回调函数是一组在训练的特定阶段被调用的函数集，你可以使用回调函数来观察训练过程中网络内部的状态和统计信息。通过传递回调函数列表到模型的.fit()中，即可在给定的训练阶段调用该函数集中的函数。
