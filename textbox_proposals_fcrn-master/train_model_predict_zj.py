#coding=utf-8
from __future__ import print_function

import itertools
import h5py
import numpy as np
import os
import random
import sys
import cv2
import scipy.misc
import math
from scipy.misc import imresize
from random import shuffle

from keras import backend as K
import theano.tensor as T
import theano
#theano.config.device = 'gpu'
#theano.config.floatX = 'float32'
import keras.models
import keras.callbacks

from theano.compile.sharedvalue import shared
from keras.models import Sequential, Model
from keras.datasets import mnist
from keras.layers import Dense, ZeroPadding2D, Activation#, Dropout,, Flatten, Input, 
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization
from keras.optimizers import SGD
from keras.utils import np_utils

img_rows = 512
img_cols = 512
nb_epoch = 100 #1000
iteration_size = 100000
mini_batch_size = 10#10
delta = 16
initial_discount = 0.01
discount_step = 1.0/90000

num_samples_per_epoch = 900#50000 
num_validation_samples = 900#5000

d = shared(initial_discount, name = 'd')

def fcrn_loss(y_true, y_pred):
  loss = K.square(y_pred - y_true)  
  images = []
  
  for i in range(0, mini_batch_size):
    c_true = y_true[i, 6, :,:].reshape((1, delta, delta))   # The last feature map in the true vals is the 'c' matrix

    c_discounted = T.set_subtensor(c_true[(c_true<=0.0).nonzero()], d.get_value())
    
    final_c = (c_discounted * loss[i,6,:,:])
       
    # Element-wise multiply of the c feature map against all feature maps in the loss
    final_loss_parts = [(c_true * loss[i, j, :, :].reshape((1, delta, delta))).reshape((1, delta, delta)) for j in range(0, 6)]
    final_loss_parts.append(final_c)
    
    images.append(K.concatenate(final_loss_parts))
    
    '''   
    final_loss_parts_new = np.array([(c_true * loss[i, j, :, :].reshape((1, delta, delta))).reshape((1, delta, delta)) for j in range(0, 6)])
    final_loss_parts_new =final_loss_parts_new*5
    final_loss_parts = list(final_loss_parts_new)
    
    final_loss_parts.append(final_c)
    images.append(K.concatenate(final_loss_parts)) '''

    #mm=K.eval(tt)
    #print (mm.shape)
    #print (K.shape(tt)) #Subtensor{::}.0
  return K.mean(K.concatenate(images).reshape((mini_batch_size, 7, delta, delta)))#, axis = 1)
 
def get_boxes(u,v,w,h):
    tl = [u-w/2, v-h/2]
    tr = [u+w/2, v-h/2]
    bl = [u-w/2, v+h/2]
    br = [u+w/2, v+h/2]
    return (tl,tr,bl,br)

def fcrn_loss_new1(y_true, y_pred):
    print ("————————————————————————————————————")
    images = []   
    final_loss_parts = []
    
   
    u_index = np.ones((16,16),dtype=np.int) 
    for j in range(0, 16): 
        u_index[:,j] = u_index[:,j]*j#*32+16    
    ku = K.variable(value=u_index, dtype='int8', name=None) #shared(u_index)
    v_index = np.ones((16,16),dtype=np.int) 
    for j in range(0, 16): 
        v_index[j,:] = v_index[j,:]*j#*32+16    
    kv = K.variable(value=v_index, dtype='int8', name=None) #shared(u_index)
    #print (K.eval(kv))
    
    '''c_discounted = T.set_subtensor(ku[(ku<i).nonzero()], 0)
    #print (K.eval(c_discounted))
    c_discounted = T.set_subtensor(c_discounted[(ku>i).nonzero()], 0)
    print (K.eval(ku))
    
    print (K.eval(c_discounted))
    print (c_discounted)'''
    #print (type(ku))
    
    for i in range(0, mini_batch_size):
        c_true = y_true[i, 6, :,:].reshape((delta, delta))   # The last feature map in the true vals is the 'c' matrix
        c_pred = y_pred[i, 6, :,:].reshape((delta, delta))
        #print ("c_true:{}\n ".format(c_true.shape))
        
        u_true = y_true[i, 0, :,:].reshape((delta, delta))
        u_pred = y_pred[i, 0, :,:].reshape((delta, delta))
        v_true = y_true[i, 1, :,:].reshape((delta, delta))   
        v_pred = y_pred[i, 1, :,:].reshape((delta, delta))
        w_true = y_true[i, 2, :,:].reshape((delta, delta))   
        w_pred = y_pred[i, 2, :,:].reshape((delta, delta))
        h_true = y_true[i, 3, :,:].reshape((delta, delta))   
        h_pred = y_pred[i, 3, :,:].reshape((delta, delta))
        
        #regression to src image
        for j in range(0, 16):    
            #t = T.set_subtensor(t[j,:], t[j,:]*32 + j*32+16)
            x_index = T.set_subtensor(ku[(ku<j).nonzero()], 0)
            x_index = T.set_subtensor(x_index[(ku>j).nonzero()], 0)
            u_true = T.set_subtensor(u_true[(x_index>0).nonzero()], u_true[(x_index>0).nonzero()]*32 + j*32+16)
            
            y_index = T.set_subtensor(kv[(kv<j).nonzero()], 0)
            y_index = T.set_subtensor(y_index[(kv>j).nonzero()], 0)
            v_true = T.set_subtensor(v_true[(y_index>0).nonzero()], v_true[(y_index>0).nonzero()]*32 + j*32+16)
        
        w_true = K.exp(w_true)*32
        w_pred = K.exp(w_pred)*32
        h_true = K.exp(h_true)*32
        h_pred = K.exp(h_pred)*32
        
        cos_true = y_true[i, 4, :,:].reshape((1, delta, delta))  
        cos_pred = y_pred[i, 4, :,:].reshape((1, delta, delta))
        sin_true = y_true[i, 5, :,:].reshape((1, delta, delta))  
        sin_pred = y_pred[i, 5, :,:].reshape((1, delta, delta))
        
        # square rect(true、pred)
        tl_square_true = [u_true-w_true/2, v_true-h_true/2]
        tr_square_true = [u_true+w_true/2, v_true-h_true/2]
        bl_square_true = [u_true-w_true/2, v_true+h_true/2]
        br_square_true = [u_true+w_true/2, v_true+h_true/2]
              
        tl_square_pred = [u_pred-w_pred/2, v_pred-h_pred/2]
        tr_square_pred = [u_pred+w_pred/2, v_pred-h_pred/2]
        bl_square_pred = [u_pred-w_pred/2, v_pred+h_pred/2]
        br_square_pred = [u_pred+w_pred/2, v_pred+h_pred/2]
        
        # Slant Rectangle (true、pred)
        tl_slant_true = [tl_square_true[0]*cos_true-tl_square_true[1]*sin_true, tl_square_true[0]*sin_true+tl_square_true[1]*cos_true]
        tr_slant_true = [tr_square_true[0]*cos_true-tr_square_true[1]*sin_true, tr_square_true[0]*sin_true+tr_square_true[1]*cos_true]
        bl_slant_true = [bl_square_true[0]*cos_true-bl_square_true[1]*sin_true, bl_square_true[0]*sin_true+bl_square_true[1]*cos_true]
        br_slant_true = [br_square_true[0]*cos_true-br_square_true[1]*sin_true, br_square_true[0]*sin_true+br_square_true[1]*cos_true]
              
        tl_slant_pred = [tl_square_pred[0]*cos_pred-tl_square_pred[1]*sin_pred, tl_square_pred[0]*sin_pred+tl_square_pred[1]*cos_pred]
        tr_slant_pred = [tr_square_pred[0]*cos_pred-tr_square_pred[1]*sin_pred, tr_square_pred[0]*sin_pred+tr_square_pred[1]*cos_pred]
        bl_slant_pred = [bl_square_pred[0]*cos_pred-bl_square_pred[1]*sin_pred, bl_square_pred[0]*sin_pred+bl_square_pred[1]*cos_pred]
        br_slant_pred = [br_square_pred[0]*cos_pred-br_square_pred[1]*sin_pred, br_square_pred[0]*sin_pred+br_square_pred[1]*cos_pred]
        
        #get center point  
        x_true_min = K.minimum(K.minimum(K.minimum(tl_slant_true[0],tr_slant_true[0]),bl_slant_true[0]),br_slant_true[0])
        x_true_max = K.maximum(K.maximum(K.maximum(tl_slant_true[0],tr_slant_true[0]),bl_slant_true[0]),br_slant_true[0])
        y_true_min = K.minimum(K.minimum(K.minimum(tl_slant_true[1],tr_slant_true[1]),bl_slant_true[1]),br_slant_true[1])
        y_true_max = K.maximum(K.maximum(K.maximum(tl_slant_true[1],tr_slant_true[1]),bl_slant_true[1]),br_slant_true[1])
        center_true = [(x_true_min+x_true_max)/2, (y_true_min+y_true_max)/2]
        
        x_pred_min = K.minimum(K.minimum(K.minimum(tl_slant_pred[0],tr_slant_pred[0]),bl_slant_pred[0]),br_slant_pred[0])
        x_pred_max = K.maximum(K.maximum(K.maximum(tl_slant_pred[0],tr_slant_pred[0]),bl_slant_pred[0]),br_slant_pred[0])
        y_pred_min = K.minimum(K.minimum(K.minimum(tl_slant_pred[1],tr_slant_pred[1]),bl_slant_pred[1]),br_slant_pred[1])
        y_pred_max = K.maximum(K.maximum(K.maximum(tl_slant_pred[1],tr_slant_pred[1]),bl_slant_pred[1]),br_slant_pred[1])
        center_pred = [(x_pred_min+x_pred_max)/2, (y_pred_min+y_pred_max)/2]
        
        dx_true = center_true[0] - u_true
        dy_true = center_true[1] - v_true
        
        dx_pred = center_pred[0] - u_pred
        dy_pred = center_pred[1] - v_pred
        
        # move
        tl_true = [tl_slant_true[0]-dx_true, tl_slant_true[1]-dy_true]
        tr_true = [tr_slant_true[0]-dx_true, tr_slant_true[1]-dy_true]
        bl_true = [bl_slant_true[0]-dx_true, bl_slant_true[1]-dy_true]
        br_true = [br_slant_true[0]-dx_true, br_slant_true[1]-dy_true]
        
        tl_pred = [tl_slant_pred[0]-dx_pred, tl_slant_pred[1]-dy_pred]
        tr_pred = [tr_slant_pred[0]-dx_pred, tr_slant_pred[1]-dy_pred]
        bl_pred = [bl_slant_pred[0]-dx_pred, bl_slant_pred[1]-dy_pred]
        br_pred = [br_slant_pred[0]-dx_pred, br_slant_pred[1]-dy_pred]
        
        # Calculation loss (coord 、 confideres)
        loss_coord = 0.01*(K.abs(tl_pred[0]-tl_true[0]) + K.abs(tl_pred[1]-tl_true[1]) +
                    K.abs(tr_pred[0]-tr_true[0]) + K.abs(tr_pred[1]-tr_true[1]) +
                    K.abs(bl_pred[0]-bl_true[0]) + K.abs(bl_pred[1]-bl_true[1]) +
                    K.abs(br_pred[0]-br_true[0]) + K.abs(br_pred[1]-br_true[1]))
        noobject_index = T.set_subtensor(c_true[(c_true<=0.0).nonzero()], 0.0)
        loss_coord = loss_coord * noobject_index
             
        loss_c = K.abs(c_pred - c_true)
        loss_c = T.set_subtensor(loss_c[(c_true<=0.0).nonzero()], loss_c[(c_true<=0.0).nonzero()]*0.5)
        
        loss = loss_coord + loss_c
        final_loss_parts.append(loss_c)
        images.append(K.concatenate(final_loss_parts))
     
    return K.mean(K.concatenate(images))
        
    
def fcrn_loss_new(y_true, y_pred):
  
    print ("++++++++++++++++++++")
    ratio_noobj = 0.5
    loss_coord = 0
    loss_c = 0
    loss = 0
    images = []
    #temp = np.array([y_pred])
    temp = np.array([(y_pred[:, j, :, :].reshape((mini_batch_size,7, delta, delta))).reshape((mini_batch_size, 7,delta, delta)) for j in range(0, 6)])
    #temp = y_pred.eval()
    print (type(temp))
    print (len(temp.shape))
    if len(temp.shape) > 1:
        for i in range(0, mini_batch_size):
            c_true = np.array([y_true[i, 6, :,:].reshape((delta, delta))]) 
            c_pred = np.array([y_pred[i, 6, :,:].reshape((delta, delta))])
    
            u_true = np.array([y_true[i, 0, :,:].reshape((delta, delta))])
            u_pred = np.array([y_pred[i, 0, :,:].reshape((delta, delta))])
            v_true = np.array(y_true[i, 1, :,:].reshape((delta, delta)))
            v_pred = np.array(y_pred[i, 1, :,:].reshape((delta, delta)))
        
            w_true = np.array(y_true[i, 2, :,:].reshape((delta, delta)))
            w_pred = np.array(y_pred[i, 2, :,:].reshape((delta, delta)))
            h_true = np.array(y_true[i, 3, :,:].reshape((delta, delta)))
            h_pred = np.array(y_pred[i, 3, :,:].reshape((delta, delta)))
        
            cos_true = np.array(y_true[i, 4, :,:].reshape((delta, delta)))
            cos_pred = np.array(y_pred[i, 4, :,:].reshape((delta, delta)))
            sin_true = np.array(y_true[i, 5, :,:].reshape((delta, delta)))
            sin_pred = np.array(y_pred[i, 5, :,:].reshape((delta, delta)))
           
            final_loss_parts = []    
            for row in range(16):
                for col in range(16):
                    [tl,tr,bl,br] = get_boxes(u_true[row][col],v_true[col][row],w_true[col][row],h_true[col][row])
                    [tl_true,tr_true,bl_true,br_true] = get_rotate_point(tl,tr,bl,br,cos,sin)
                
                    [tl,tr,bl,br] = get_boxes(u_pred[row][col],v_pred[col][row],w_pred[col][row],h_pred[col][row])
                    [tl_pred,tr_pred,bl_pred,br_pred] = get_rotate_point(tl,tr,bl,br,cos,sin)
                
                    loss_coord = 5*(fabs(tl_pred[0]-tl_true[0]) + fabs(tl_pred[1]-tl_true[1]) +
                    fabs(tr_pred[0]-tr_true[0]) + fabs(tr_pred[1]-tr_true[1]) +
                    fabs(bl_pred[0]-bl_true[0]) + fabs(bl_pred[1]-bl_true[1]) +
                    fabs(br_pred[0]-br_true[0]) + fabs(br_pred[1]-br_true[1]))
                
                    if c_true[row][col]<0.01:
                        loss_c = 0.5*fabs(c_pred[row][col]-c_true[row][col])
                    else:
                        loss_c = fabs(c_pred[row][col]-c_true[row][col])
                
                    loss = loss + loss_coord + loss_c  
                    final_loss_parts.append(list(1000+loss))
                    
            images.append(K.concatenate(final_loss_parts))
                
        return K.mean(K.concatenate(images).reshape((mini_batch_size, 1))) 
    else:
        
        return K.mean(K.square(y_pred - y_true)) 
                                              
 
class DiscountCallback(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    print("Running callback: " + str(epoch))
    d.set_value(d.get_value() + discount_step)
    
def build_model():
    model = Sequential()
    
    # Layer 1
    model.add(ZeroPadding2D(padding = (2, 2), input_shape=(1, img_rows, img_cols))) #theano (channels,w,h) ,tensorflow (w,h,channels)
    model.add(Convolution2D(64, 5, 5))
    model.add(BatchNormalization(axis = 1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2,2)))
    print("Layer 1: " + str(model.layers[-1].output_shape))
    
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
    

    # Layer 10
    model.add(ZeroPadding2D(padding = (2, 2)))
    model.add(Convolution2D(7, 5, 5))
    model.add(BatchNormalization(axis = 1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2,2)))
    print("Layer 10: " + str(model.layers[-1].output_shape))
    
    sgd = SGD(lr = 10e-4, decay = 5e-4, momentum = 0.9, nesterov = False)
    
    model.compile(loss = fcrn_loss_new1, optimizer = sgd, metrics = ['accuracy'])
    
    return model
        
def batch(iterable, n = 1):
  current_batch = []
  #print ("batch-----------------\n")
  for item in iterable:
    current_batch.append(item)
    #print ("current_batch:{}\n".format(str(current_batch)))
    if len(current_batch) == n:
      yield current_batch
      current_batch = []
      
global a
a = 0

def Fuc():
    global a
    print ("index:{}\n".format(a))
    a = a + 1
   
def exemplar_generator(db_iters, batch_size):
  while True:
    #print ("+++++++++++++++++\n")
    #print ("db_iters:{}\n".format(db_iters))    
    db_path = "../small_data"
    dbs = map(lambda x: db_path + "/" + x, [f for f in os.listdir(db_path) if os.path.isfile(db_path + "/" + f)])
    db_iters = map(lambda x: load_db(x), dbs)
    
    for chunk in batch(itertools.chain.from_iterable(db_iters), batch_size):
      X = []
      Y = []
           
      for item in chunk:
        X.append(item[:].reshape(1, img_rows, img_cols))
        labels = np.array(item.attrs['label']).transpose(2, 0, 1)
        Y.append(labels.reshape(7, delta, delta))  
        #Fuc()
      yield (np.array(X), np.array(Y))
      
def load_db(db_filename):
  try:
    db = h5py.File(db_filename, 'r')
    return db['data'].itervalues()
  except:
    print(sys.exc_info()[1])
    return []

def get_move_roi(tl,tr,bl,br, center0,center1):
    dx = center1[0] - center0[0]
    dy = center1[1] - center0[1]
      
    tl = (tl[0]-dx, tl[1]-dy)
    tr = (tr[0]-dx, tr[1]-dy)
    bl = (bl[0]-dx, bl[1]-dy)
    br = (br[0]-dx, br[1]-dy)
    return (tl,tr,bl,br)
    
    
def get_roi_center(tl,tr,bl,br):
    x_min = min(tl[0],tr[0],bl[0],br[0])
    x_max = max(tl[0],tr[0],bl[0],br[0])
    y_min = min(tl[1],tr[1],bl[1],br[1])
    y_max = max(tl[1],tr[1],bl[1],br[1])
    return (float((x_min + x_max))/2.0, float((y_min + y_max))/2.0)
 
def get_twoRect_External(tl0,tr0,bl0,br0,tl1,tr1,bl1,br1):
    x0_min = min(tl0[0],tr0[0],bl0[0],br0[0])
    x0_max = max(tl0[0],tr0[0],bl0[0],br0[0])
    y0_min = min(tl0[1],tr0[1],bl0[1],br0[1])
    y0_max = max(tl0[1],tr0[1],bl0[1],br0[1])
    
    x1_min = min(tl1[0],tr1[0],bl1[0],br1[0])
    x1_max = max(tl1[0],tr1[0],bl1[0],br1[0])
    y1_min = min(tl1[1],tr1[1],bl1[1],br1[1])
    y1_max = max(tl1[1],tr1[1],bl1[1],br1[1])
    
    x_min = (int)(min(x0_min,x1_min))
    x_max = (int)(max(x0_max,x1_max))
    y_min = (int)(min(y0_min,y1_min))
    y_max = (int)(max(y0_max,y1_max))
    return (x_min,x_max,y_min,y_max)
    
def get_rotate_point(tl,tr,bl,br,cos,sin):
    tl = (tl[0]*cos-tl[1]*sin, tl[0]*sin+tl[1]*cos)
    tr = (tr[0]*cos-tr[1]*sin, tr[0]*sin+tr[1]*cos)
    bl = (bl[0]*cos-bl[1]*sin, bl[0]*sin+bl[1]*cos)
    br = (br[0]*cos-br[1]*sin, br[0]*sin+br[1]*cos)
    '''print ("tl:{}\n ".format(tl))
    R = math.pow(cos,2) + math.pow(sin,2)
    R = math.sqrt(R)
    print ("R:{}\n ".format(R))
    print ("cos:{}\n".format(cos))
    print ("sin:{}\n".format(sin))'''
    return (tl,tr,bl,br)

def get_cross(pt1,pt2,pt):
    return (pt2[0]-pt1[0])*(pt[1]-pt1[1]) - (pt[0]-pt1[0])*(pt2[1]-pt1[1])

def IsPointInMatrix(tl,tr,bl,br,pt):
    return get_cross(tl,bl,pt) * get_cross(br,tr,pt) >= 0 and get_cross(bl,br,pt) * get_cross(tr,tl,pt) >= 0
 
def get_twoRect_IOU(tl0,tr0,bl0,br0,tl1,tr1,bl1,br1):
    (x_min,x_max,y_min,y_max) = get_twoRect_External(tl0,tr0,bl0,br0,tl1,tr1,bl1,br1)
    num = 0
    inter_num = 0
    union_num = abs(x_max-x_min)*abs(y_max-y_min)

    for col in range(x_min,x_max):
        for row in range(y_min,y_max):
            pt = (col,row)
            if IsPointInMatrix(tl0,tr0,bl0,br0,pt) and IsPointInMatrix(tl1,tr1,bl1,br1,pt):
                inter_num = inter_num + 1
            elif (not IsPointInMatrix(tl0,tr0,bl0,br0,pt)) or (not IsPointInMatrix(tl1,tr1,bl1,br1,pt)):
                num = num + 1
    return float(inter_num)/(union_num-num+1)
                          
def nms(lst,confideres,threshold):
    if len(lst)==0:
        return []
    num = len(lst)
    #print ("nms num:{}".format(num))
    labels = np.zeros_like(confideres, dtype=np.int16)
    res = []     
    #print (confideres)
    for  i in range(num):
        print (i)
        for j in range(i+1, num):
            tl0 = lst[i][0]
            tr0 = lst[i][1]
            bl0 = lst[i][2]
            br0 = lst[i][3]
        
            tl1 = lst[j][0]
            tr1 = lst[j][1]
            bl1 = lst[j][2]
            br1 = lst[j][3]
            o = get_twoRect_IOU(tl0,tr0,bl0,br0,tl1,tr1,bl1,br1)
            #print (o)
            if o > threshold:
                if confideres[i] > confideres[j]:
                    labels[j] = -1
                else:
                    labels[i] = -1

    #print (labels)
    for i in range(num):
        if labels[i] > -1 :
            res.append(lst[i])
    return res
     
 
def draw_roi(img,res):
    '''print (len(res))
    print (len(res[0]))
    print (len(res[0][0]))
    print (len(res[0][0][0]))

    for i in range(0,len(res[0])):
       print (res[0][i][0][0])#(res[0][i])
       print ('\n')'''
    #print ("..........")
    x = res[0,0,:,:]
    y = res[0,1,:,:]
    w = res[0,2,:,:]
    h = res[0,3,:,:]
    cos = res[0,4,:,:]
    sin = res[0,5,:,:]
    c = res[0,6,:,:]
    
    boxes = []
    confideres = []
    img_src = img.copy()
    for row in range(0,16):
        for col in range(0,16):
            if w[row][col]>0.0 and h[row][col]>0.0 and c[row][col] > 0.5:
                centerX = x[row][col]*32 + col
                centerY = y[row][col]*32 + row
                ww = 32*math.exp(w[row][col])
                hh = 32*math.exp(h[row][col])
                
                tl = (centerX-ww/2, centerY-hh/2)
                tr = (centerX+ww/2, centerY-hh/2)
                bl = (centerX-ww/2, centerY+hh/2)
                br = (centerX+ww/2, centerY+hh/2)                
                center0 = get_roi_center(tl,tr,bl,br)
                
                cv2.line(img_src,(int(tl[0]),int(tl[1])),(int(tr[0]),int(tr[1])),(0,0,255),1)
                cv2.line(img_src,(int(tl[0]),int(tl[1])),(int(bl[0]),int(bl[1])),(0,0,255),1)
                cv2.line(img_src,(int(bl[0]),int(bl[1])),(int(br[0]),int(br[1])),(0,0,255),1)
                cv2.line(img_src,(int(tr[0]),int(tr[1])),(int(br[0]),int(br[1])),(0,0,255),1)
                
                #print (tl)
                (tl,tr,bl,br) = get_rotate_point(tl,tr,bl,br,cos[row][col],sin[row][col])
                center1 = get_roi_center(tl,tr,bl,br)           
                (tl,tr,bl,br) = get_move_roi(tl,tr,bl,br, center0,center1)
                
                boxes.append([tl,tr,bl,br])
                confideres.append(c[row][col]) 
                #print (tl)
                cv2.line(img,(int(tl[0]),int(tl[1])),(int(tr[0]),int(tr[1])),(0,0,255),1)
                cv2.line(img,(int(tl[0]),int(tl[1])),(int(bl[0]),int(bl[1])),(0,0,255),1)
                cv2.line(img,(int(bl[0]),int(bl[1])),(int(br[0]),int(br[1])),(0,0,255),1)
                cv2.line(img,(int(tr[0]),int(tr[1])),(int(br[0]),int(br[1])),(0,0,255),1)
                #cv2.rectangle(img, (xx,yy), (int(xx+w[row][col]*512),int(yy+h[row][col]*512)),(0,0,255),1)
                #print (x[row][col],y[row][col])
                #print (ww,hh,c[row][col])
    cv2.imwrite("test.bmp", img)
    cv2.imwrite("img_src.bmp", img_src)
    return (boxes,confideres)
    
def nms_square(boxes, threshold, method): #nms(total_boxes, 0.7, 'Union') #error
    if len(boxes)==0:
        return np.empty((0,3))
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    s = boxes[:,4]
    area = (x2-x1+1) * (y2-y1+1)
    I = np.argsort(s)
    pick = np.zeros_like(s, dtype=np.int16)
    counter = 0
    while I.size>0:
        i = I[-1]
        pick[counter] = i
        counter += 1
        idx = I[0:-1]
        xx1 = np.maximum(x1[i], x1[idx])
        yy1 = np.maximum(y1[i], y1[idx])
        xx2 = np.minimum(x2[i], x2[idx])
        yy2 = np.minimum(y2[i], y2[idx])
        w = np.maximum(0.0, xx2-xx1+1)
        h = np.maximum(0.0, yy2-yy1+1)
        inter = w * h
        if method is 'Min':
            o = inter / np.minimum(area[i], area[idx])
        else:
            o = inter / (area[i] + area[idx] - inter)
        I = I[np.where(o<=threshold)]
    pick = pick[0:counter]
    return pick
    
def get_squareBoxes(boxes): 
    x = res[0,0,:,:]
    y = res[0,1,:,:]
    w = res[0,2,:,:]
    h = res[0,3,:,:]
    c = res[0,4,:,:]
    
    boxes = []
    confideres = []
    for row in range(0,16):
        for col in range(0,16):
            if w[row][col]>0.0 and h[row][col]>0.0 and c[row][col] > 0.00001:
                centerX = x[row][col]*32 + col*32+16
                centerY = y[row][col]*32 + row*32+16
                ww = 32*math.exp(w[row][col])
                hh = 32*math.exp(h[row][col])
                
                tl = (centerX-ww/2, centerY-hh/2)
                tr = (centerX+ww/2, centerY-hh/2)
                bl = (centerX-ww/2, centerY+hh/2)
                br = (centerX+ww/2, centerY+hh/2)                
                
                print (ww,hh, c[row][col])
                if ww>20 and hh>20:
                    boxes.append([tl,tr,bl,br])
                    confideres.append(c[row][col]) 
                          
    return (boxes,confideres)   
 
def draw_squareBoxes(img, boxes, confideres, threshold): 
    print ("final boxes num:{}\n".format(len(boxes)))
    if len(boxes) < 1:
        return

    #final_boxes = nms_square(boxes, threshold, 'Union')    
    for i in range(len(boxes)):
        [tl,tr,bl,br] = boxes[i][:]
        if confideres[i] > 0.0001:#0.4:
            s = str(round(confideres[i],2))
            cv2.putText(img, s, (int((tl[0]+br[0])/2),int((tl[1]+br[1])/2)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0 ,0), thickness = 1, lineType = 8) 
            cv2.rectangle(img,(int(tl[0]),int(tl[1])),(int(br[0]),int(br[1])),(0,0,255),1)        
    cv2.imwrite("result.bmp", img)                  

def ger_realBox(res):
    x = res[0,0,:,:]
    y = res[0,1,:,:]
    w = res[0,2,:,:]
    h = res[0,3,:,:]
    cos = res[0,4,:,:]
    sin = res[0,5,:,:]
    c = res[0,6,:,:]
    
    #print (c)
    boxes = []
    confideres = []
    for row in range(0,16):
        for col in range(0,16):
            if w[row][col]>0.0 and h[row][col]>0.0 and c[row][col] >0.05:
                centerX = x[row][col]*32 + col*32+16
                centerY = y[row][col]*32 + row*32+16
                ww = 32*math.exp(w[row][col])
                hh = 32*math.exp(h[row][col])
                
                tl = (centerX-ww/2, centerY-hh/2)
                tr = (centerX+ww/2, centerY-hh/2)
                bl = (centerX-ww/2, centerY+hh/2)
                br = (centerX+ww/2, centerY+hh/2)                
                center0 = [centerX, centerY]
                print (ww,hh)
                (tl,tr,bl,br) = get_rotate_point(tl,tr,bl,br,cos[row][col],sin[row][col])
                center1 = get_roi_center(tl,tr,bl,br)           
                (tl,tr,bl,br) = get_move_roi(tl,tr,bl,br, center0,center1)
                
                if ww>30 and hh>30:             
                    boxes.append([tl,tr,bl,br])
                    confideres.append(c[row][col]) 
               
    return (boxes,confideres)   
    
def draw_boxes(img, boxes): 
    print (len(boxes))
    for i in range(len(boxes)):
        [tl,tr,bl,br] = boxes[i]
        cv2.line(img,(int(tl[0]),int(tl[1])),(int(tr[0]),int(tr[1])),(0,0,255),1)
        cv2.line(img,(int(tl[0]),int(tl[1])),(int(bl[0]),int(bl[1])),(0,0,255),1)
        cv2.line(img,(int(bl[0]),int(bl[1])),(int(br[0]),int(br[1])),(0,0,255),1)
        cv2.line(img,(int(tr[0]),int(tr[1])),(int(br[0]),int(br[1])),(0,0,255),1)
        center = get_roi_center(tl,tr,bl,br)
        print (tl,tr,bl,br)
        s = str(i)
        cv2.putText(img,s, (int(center[0]),int(center[1])), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0 ,0), thickness = 1, lineType = 8)  
    cv2.imwrite("result.bmp", img)
   
    
def load_exemplars(db_path):
  dbs = map(lambda x: db_path + "/" + x, [f for f in os.listdir(db_path) if os.path.isfile(db_path + "/" + f)])
  print ("load_exemplars ...........")
  return exemplar_generator(map(lambda x: load_db(x), dbs), mini_batch_size)

if __name__ == '__main__':
  model_file = "bb-fcrn-model_weight_newLoss"
  train_db_path = "../small_data" #"/path/to/dbs"
  validate_db_path = "../small_data" #"/path/to/dbs"
  
  print("Loading data...")
  
  train = load_exemplars(train_db_path)
  validate = load_exemplars(validate_db_path)
  
  print("Data loaded.")
  print("Building model...")
  
  model = build_model()
  
  checkpoint = keras.callbacks.ModelCheckpoint(model_file + ".h5",
                                               monitor = "acc",
                                               verbose = 1,
                                               save_best_only = True,
                                               save_weights_only = True,
                                               mode = 'auto')
  
  earlystopping = keras.callbacks.EarlyStopping(monitor = 'loss',
                                                min_delta = 0,
                                                patience = 5,
                                                verbose = 1,
                                                mode = 'auto')
  
  discount = DiscountCallback()
  
  csvlogger = keras.callbacks.CSVLogger(model_file + "-log.csv", append = True)
  
      
  if os.path.exists(model_file + ".h5"):
    model.load_weights(model_file + ".h5")
    print ("load weights ok!")
    
    #read train image as test image
    h5_Imagepath = "../small_data/JPLMC3"
    i = 10

    if os.path.exists(h5_Imagepath): 
        dbs = h5py.File(h5_Imagepath, 'r')
        data = dbs['data']

        #print ( len(dbs) ) #1 0CZS0K
        #print (len(data)) #1000
        key = data.keys()
        img = data[key[i]][:]
        cv2.imwrite("test_src.bmp", img)
        #print (data[key[0]].shape)
        #print (data[key[0]].attrs['label']) #16*16*7
        #print (type(data))
    
    image_path = "./test_src.bmp"
    if os.path.exists(image_path):
        img = cv2.imread(image_path)
    else:
        print ("Image path not found!")
        
    if img == None:
        os._exit() 
    print (img.shape)
    h_scale = img_rows / float(img.shape[0])
    w_scale = img_cols / float(img.shape[1])
          
    img_color = imresize(img, (int(img_rows), int(img_cols)), interp = 'bicubic')
    img = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY)
    img1 = np.expand_dims(img, axis=0) #扩一维
    img1 = np.expand_dims(img1, axis=0)
    #    
      
    print ("start predict!")      
    res = model.predict(img1) #predict_on_batch(np.array(train.next()[0])) #1*7*16*16 (batch_size*7*16*16)
    print ("predict over!")
    print ("predict boxes num:{}\n".format(len(res)))
    
    if 1:       
        [boxes,confideres] = ger_realBox(res)
        print ("nms before boxes num:{}\n".format(len(boxes)))
        result = nms(boxes,confideres,0.2)
        print ("nms after boxes num:{}\n".format(len(result)))
        draw_boxes(img, result)
    else:
        [boxes, confideres] = get_squareBoxes(res)
        draw_squareBoxes(img, boxes, confideres, 0.4)
       
  else:
    model.fit_generator(train,
                      samples_per_epoch = num_samples_per_epoch,
                      nb_epoch = nb_epoch,
                      verbose = 1,
                      validation_data = validate,
                      nb_val_samples = num_validation_samples,
                      max_q_size = 10,
                      pickle_safe = True,
                      callbacks = [checkpoint, earlystopping, csvlogger, discount])
                      
                     
