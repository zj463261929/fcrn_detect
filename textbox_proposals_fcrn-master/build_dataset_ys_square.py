from math import ceil
import scipy.misc
from scipy.misc import imresize
import cv2
import sys
import os
import os.path
import string
import h5py
import numpy as np
from collections import namedtuple
import random
import time
import math
import traceback
import scipy.io as sio

delta = 16.0
W = 512.0
H = 512.0
max_db_size = 1000

def get_maxRect(tl,tr,bl,br):
    x_min = min(tl[0],tr[0],bl[0],br[0])
    x_max = max(tl[0],tr[0],bl[0],br[0])
    y_min = min(tl[1],tr[1],bl[1],br[1])
    y_max = max(tl[1],tr[1],bl[1],br[1])
    
    u = (x_min+x_max)/2
    v = (y_min+y_max)/2
    w = math.fabs(x_max - x_min)
    h = math.fabs(y_max - y_min)
    
    # find midpoint
    (x,y) = (float((x_min + x_max)) / 2.0, float((y_min + y_max) / 2.0))

    # calculate U,V
    cell_W = W / delta
    cell_H = H / delta
    
    (u, v) = (math.floor(x / cell_W) * cell_W + (cell_W / 2.0), math.floor(y / cell_H) * cell_H + (cell_H / 2.0))
    
    tl = [x_min,y_min]
    tr = [x_max, y_min]
    bl = [x_min, y_max]
    br = [x_max, y_max]
    #print ("tl,tr,bl,br,w,h:{}\n".format((tl[:],tr[:],bl[:],br[:],w,h)))
    if w>0.00001 and h>0.00001:
        return (tl,tr,bl,br, (x-u) / cell_W, (y-v) / cell_H, math.log(float(w) / cell_W), math.log(float(h) / cell_H),1)
    else:
        return (0,0,0,0,0, 0, 0, 0, 0)
    
def get_rotation(tl, tr):
    defs = (float(tr[0] - tl[0]), float(tr[1] - tl[1]))

    rotation = math.atan2(defs[1], defs[0]) * 180.0 / math.pi

    if defs[1] < 0:
        rotation += 180

    elif defs[0] < 0:
        rotation += 360

    return rotation

def calc_pose(tl, tr, bl, br):
  
    x_min = min(tl[0],tr[0],bl[0],br[0])
    x_max = max(tl[0],tr[0],bl[0],br[0])
    y_min = min(tl[1],tr[1],bl[1],br[1])
    y_max = max(tl[1],tr[1],bl[1],br[1])
    
    # find midpoint
    #(x, y) = (float((tl[0] + br[0])) / 2.0, float((tl[1] + br[1]) / 2.0))
    (x,y) = (float((x_min + x_max)) / 2.0, float((y_min + y_max) / 2.0))

    # calculate U,V
    cell_W = W / delta
    cell_H = H / delta
    #print cell_W
    #print cell_H
    
    (u, v) = (math.floor(x / cell_W) * cell_W + (cell_W / 2.0), math.floor(y / cell_H) * cell_H + (cell_H / 2.0))
    #print(u,v)
    #print(x,y)
    
    # Calculate theta
    theta = get_rotation(tl, tr)

    h1 = math.sqrt(math.pow((bl[0]-tl[0]),2) + math.pow((bl[1]-tl[1]),2))
    h2 = math.sqrt(math.pow((br[0]-tr[0]),2) + math.pow((br[1]-tr[1]),2))
    h = (h1+h2)/2   
    w1 = math.sqrt(math.pow((tr[0]-tl[0]),2) + math.pow((tr[1]-tl[1]),2))
    w2 = math.sqrt(math.pow((bl[0]-br[0]),2) + math.pow((bl[1]-br[1]),2))
    w = (w1+w2)/2
    
    #print ("w1,w2,h1,h2,w,h:{}\n".format((w1,w2,h1,h2,w,h)))
    
    #w = tr[0] - tl[0]
    #h = bl[1] - tl[1]
    
    #print ("tl,tr,bl,br,w,h:{}\n".format((tl[:],tr[:],bl[:],br[:],w,h)))
    if w>0.00001 and h>0.00001:
        return ((x-u) / cell_W, (y-v) / cell_H, math.log(float(w) / cell_W), math.log(float(h) / cell_H), math.cos(theta), math.sin(theta), 1)
    else:
        return (0, 0, 0, 0, 0, 0, 0)
        
def id_generator(size = 6, chars = string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

def create_new_db(path):
  
    filename = path + "/" + id_generator()
    while os.path.exists(filename):
        filename = path + "/" + id_generator()    

    db = h5py.File(filename, 'w')
    db.create_group("/data")

    print("Created DB: " + filename)
    return db

def add_res_to_db(db, img, labels):
    try:
        seed = id_generator() + "_" + str(int(round(time.time() * 1000)))
        data = img

        db['data'].create_dataset(seed, data = data)
        db['data'][seed].attrs['label'] = np.array(labels)
    except:
        print(sys.exc_info()[1])
    
def generate_dataset(db_location, output_location):
    if not os.path.exists(output_location):
        os.makedirs(output_location)
    
    dbs = [f for f in os.listdir(db_location) if os.path.isfile(db_location + "/" + f)]

    random.shuffle(dbs)

    total = 0
    images = 0

    out_db = create_new_db(output_location)

    for cur_db in dbs:
    
        try:
            print(db_location + "/" + cur_db)
            in_db = sio.loadmat(db_location + "/" + cur_db)            
            count = 0
            '''print in_db.keys() 
            print in_db['wordBB'][0]
            print in_db['wordBB'][0][:]
            print len(in_db['wordBB'][0])
            print in_db['wordBB'][0][1]
            print len(in_db['wordBB'][0][1])
            print in_db['txt'][0][1]
            print in_db['wordBB'][0][2]
            print in_db['wordBB'][0][2].shape
            print in_db['txt'][0][2]'''
            
            imageNum = len(in_db['imnames'][0])
            print imageNum          
            #imageNum = 1
            for index in np.arange(imageNum):
                ss = str(in_db['imnames'][0][index])
                print ss[3:len(ss)-2]
                
                '''s2 = ss[3:len(ss)-2]
                if s2 != '182/turtles_44_14.jpg': #'8/ballet_131_35.jpg':
                   continue'''
   
                strqq = db_location + "/" + ss[3:len(ss)-2]
                print strqq
            
                #img = item[:].astype('float32')
                #print ("image path:{}\n".format(strqq))
                               
                img = cv2.imread(strqq)
                image_src = img.copy()
                
                if img!=None:
                
                    orig_dims = img.shape

                    h_scale = H / float(img.shape[0])
                    w_scale = W / float(img.shape[1])
                    #print (img.shape)
                    img = imresize(img, (int(H), int(W)), interp = 'bicubic')
                    image_color = img.copy()
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

                    images +=1 
                    print ("image num:{}\n".format(images))

                    # BBs and word labels are both lists where corresponding indices match
                    wordBB = in_db['wordBB'][0][index]
                
                    labels = np.empty((16, 16, 5), dtype = 'float64')

                    h_step = H / delta
                    w_step = W / delta

                    # Loop through each of the segments and determine labels
                    for i in range(0, int(delta)):
                
                        minX = i * h_step
                        maxX = i * h_step + h_step
                
                        for j in range(0, int(delta)):
                  
                            minY = j * w_step
                            maxY = j * w_step + w_step

                            # if the center point of a bounding box lies within the given segment, calculat the pose info
                            # otherwise, set all pose to 0

                            (x, y, w, h, cos, sin) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                            c = 0.0

                            # Loop through labels and put in proper directories
                            #print wordBB.shape                      
                            flag = 0
                            if len(wordBB.shape) == 3:  
                                wordBBNum = wordBB.shape[-1]                          
                                for i2 in xrange(wordBB.shape[-1]):                                 
                                    bb = wordBB[:,:,i2]                            
                                    bb = np.c_[bb, bb[:,0]]                            
                                    
                                    (tl, tr, br, bl) = bb[0:, 0:4].T
                                    '''
                                    #cv2.rectangle(image_src,(int(tl[0]),int(tl[1])),(int(br[0]),int(br[1])),(0,0,255),1)
                                    cv2.line(image_src,(int(tl[0]),int(tl[1])),(int(tr[0]),int(tr[1])),(0,0,255),1)
                                    cv2.line(image_src,(int(tl[0]),int(tl[1])),(int(bl[0]),int(bl[1])),(0,0,255),1)
                                    cv2.line(image_src,(int(bl[0]),int(bl[1])),(int(br[0]),int(br[1])),(0,0,255),1)
                                    cv2.line(image_src,(int(tr[0]),int(tr[1])),(int(br[0]),int(br[1])),(0,0,255),1)'''
                                    
                                    #print ("src:tl,tr,bl,br:{}\n".format((tl[:],tr[:],bl[:],br[:])))
                                    tl = (tl[0]*w_scale, tl[1]*h_scale)
                                    tr = (tr[0]*w_scale, tr[1]*h_scale)
                                    br = (br[0]*w_scale, br[1]*h_scale)
                                    bl = (bl[0]*w_scale, bl[1]*h_scale)
                                    
                                    [tl,tr,bl,br,u,v,w,h,c] = get_maxRect(tl,tr,bl,br)
                                    '''
                                    s1 = '('+str(int(tl[0])) + ',' + str(int(tl[1]))+')'
                                    cv2.putText(image_color, s1, (int((tl[0]+br[0])/2),int((tl[1]+br[1])/2)), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0 ,0), thickness = 1, lineType = 8)  
                                    #cv2.rectangle(image_color,(int(tl[0]),int(tl[1])),(int(br[0]),int(br[1])),(0,0,255),1)
                                    cv2.line(image_color,(int(tl[0]),int(tl[1])),(int(tr[0]),int(tr[1])),(0,0,255),1)
                                    cv2.line(image_color,(int(tl[0]),int(tl[1])),(int(bl[0]),int(bl[1])),(0,0,255),1)
                                    cv2.line(image_color,(int(bl[0]),int(bl[1])),(int(br[0]),int(br[1])),(0,0,255),1)
                                    cv2.line(image_color,(int(tr[0]),int(tr[1])),(int(br[0]),int(br[1])),(0,0,255),1)'''
                                                                       
                                    # if the midpoint of the current ground truth BB is within the current segment, calculate pose params
                                    if u >= minX and u <= maxX and v >= minY and v <= maxY:
                                        c = 1.0
                                        flag = 1
                                                  
                                #cv2.imwrite("data.bmp", image_color)
                                #cv2.imwrite("src.bmp", image_src)
                            else:
                                wordBBNum = 1
                                bb = wordBB                            
                                bb = np.c_[bb, bb[:,0]]                            
                                
                                (tl, tr, br, bl) = bb[0:, 0:4].T
                                tl = (tl[0]*w_scale, tl[1]*h_scale)
                                tr = (tr[0]*w_scale, tr[1]*h_scale)
                                br = (br[0]*w_scale, br[1]*h_scale)
                                bl = (bl[0]*w_scale, bl[1]*h_scale)
                                
                                [tl,tr,bl,br,u,v,w,h,c] = get_maxRect(tl,tr,bl,br)
                                
                                # if the midpoint of the current ground truth BB is within the current segment, calculate pose params
                                if u >= minX and u <= maxX and u >= minY and u <= maxY:
                                    c = 1.0
                                    flag = 1
                            if flag==1: 
                                #print (x,y,w,h,sin,cos,c)                            
                                labels[i][j] = np.array([u, v, w, h, c])
                                #print ("-------------")
                  
                    count += 1
                    total +=1

                    if total == max_db_size:
                        total = 0
                        out_db = create_new_db(output_location)
                
                
                    # This is a bug in the synthtext dataset. It's possible there are bounding boxes with boundaries outside the actual image
                    if np.count_nonzero(labels[:,:,4]) == 0:
                        print("FOUND NONZERO- hscale=" + str(h_scale) + ", wscale=" + str(w_scale) + ", size=" + str(orig_dims))
                        print(str(wordBB.T))

                    else:
                        add_res_to_db(out_db, img, labels)
            
            print("Extracted " + str(count) + " images from " + cur_db + ".")
        
        except:
      
            print("Error loading from " + cur_db + "(" + str(sys.exc_info()[0]) + ") continuing...")
            print(str(sys.exc_info()[1]))
            traceback.print_tb(sys.exc_info()[2])
      
    print("Total number of ground truth images: " + str(images))

if __name__ == "__main__":
    db_dir = sys.argv[1]
    output_dir = sys.argv[2]

    generate_dataset(db_dir, output_dir)
