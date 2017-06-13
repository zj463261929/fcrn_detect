#coding=utf-8
import math
import cv2
import os
import h5py
import numpy as np
 
def lst_square():
    lst1 = []
 
 
 
def test_h5():
    h5_path = "../h5/JPLMC3" #"../small_data/0CZS0K"
    t = 0
    if os.path.exists(h5_path): 
        dbs = h5py.File(h5_path, 'r')
        data = dbs['data']

        print ( len(dbs) ) #1 0CZS0K
        print (len(data)) #1000
        key = data.keys()
        for i in range(10):
            img = data[key[i]][:]
            s = "./data/" + str(t) + ".bmp"
            cv2.imwrite(s, img)
            t = t+1
            print (data[key[0]].shape)
            print (t)
            #print (data[key[0]].attrs['label']) #16*16*7
            print (type(data))

#test_h5() 
 
def get_cross(pt1,pt2,pt):
    return (pt2[0]-pt1[0])*(pt[1]-pt1[1]) - (pt[0]-pt1[0])*(pt2[1]-pt1[1])

def IsPointInMatrix(tl,tr,bl,br,pt):
    return get_cross(tl,bl,pt) * get_cross(br,tr,pt) >= 0 and get_cross(bl,br,pt) * get_cross(tr,tl,pt) >= 0
 
def get_twoRect_External(tl0,tr0,bl0,br0,tl1,tr1,bl1,br1):
    x0_min = min(tl0[0],tr0[0],bl0[0],br0[0])
    x0_max = max(tl0[0],tr0[0],bl0[0],br0[0])
    y0_min = min(tl0[1],tr0[1],bl0[1],br0[1])
    y0_max = max(tl0[1],tr0[1],bl0[1],br0[1])
    
    x1_min = min(tl1[0],tr1[0],bl1[0],br1[0])
    x1_max = max(tl1[0],tr1[0],bl1[0],br1[0])
    y1_min = min(tl1[1],tr1[1],bl1[1],br1[1])
    y1_max = max(tl1[1],tr1[1],bl1[1],br1[1])
    
    x_min = min(x0_min,x1_min)
    x_max = max(x0_max,x1_max)
    y_min = min(y0_min,y1_min)
    y_max = max(y0_max,y1_max)
    return (x_min,x_max,y_min,y_max)
    
    
def get_twoRect_IOU(tl0,tr0,bl0,br0,tl1,tr1,bl1,br1):
    (x_min,x_max,y_min,y_max) = get_twoRect_External(tl0,tr0,bl0,br0,tl1,tr1,bl1,br1)
    union_num = 0
    intersection_num = 0
    
    for col in range(x_min,x_max):
        for row in range(y_min,y_max):
            pt = (col,row)
            if IsPointInMatrix(tl0,tr0,bl0,br0,pt) and IsPointInMatrix(tl1,tr1,bl1,br1,pt):
                intersection_num = intersection_num + 1
            elif (not IsPointInMatrix(tl0,tr0,bl0,br0,pt)) or (not IsPointInMatrix(tl1,tr1,bl1,br1,pt)):
                union_num = union_num + 1
    return float(intersection_num)/union_num

def nms(lst,confideres,threshold):
    if len(lst)==0:
        return []
    num = len(lst)
    
    labels = np.zeros_like(confideres, dtype=np.int16)
    res = []     
    print (confideres)
    for  i in range(num):
        I = []
        lst_row = []
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
            print (o)
            if o > threshold:
                if confideres[i] > confideres[j]:
                    labels[j] = -1
                else:
                    labels[i] = -1

    print (labels)
    for i in range(num):
        if labels[i] > -1 :
            res.append(lst[i])
    return res
    
    ''' arr = np.array(lst_IOU)    
    I = np.where(arr<=threshold)#小于阈值的位置
    print (I)
    for i in range(len(lst_IOU)):
        if lst_IOU[i] < threshold:
            res.append(lst[labels[i][0]])
            res.append(lst[labels[i][1]])
        elif lst_IOU[i] > threshold:
            if confideres[labels[i][0]] < confideres[labels[i][1]]:
                res.append(lst[labels[i][1]])
            else:
                res.append(lst[labels[i][0]])
       
    return res'''
'''
def nms1(boxes, confideres, overlap):
 
# Non-maximum suppression.
# In object detect algorithm, select high score detections and skip windows
# covered by a previously selected detection.
#
# input - boxes : object detect windows.
#                 xMin yMin xMax yMax score.
#         overlap : suppression threshold.
# output - pickLocate : number of local maximum score.
 
pickLocate = [] 

if len(lst)==0:
    return pickLocate
else         
    # sort detected windows based on the score.
    I = np.argsort(confideres)
     
    pickLocate = []
    counter = 0
    while I.size>0:
        i = I[-1]
         
        pickLocate[counter] = i
        counter += 1
        suppress = [-1];
         
        for pos in range(len(boxs)):
            j = I[pos] 
             
            tl0 = boxes[i][0]
            tr0 = boxes[i][1]
            bl0 = boxes[i][2]
            br0 = boxes[i][3]
        
            tl1 = boxes[j][0]
            tr1 = boxes[j][1]
            bl1 = boxes[j][2]
            br1 = boxes[j][3]
               
            o = get_twoRect_IOU(tl0,tr0,bl0,br0,tl1,tr1,bl1,br1)
                 
            if o > overlap:
                suppress = [suppress; pos];

def nms2(boxes, confideres, threshold):
    if boxes.size==0:
        return np.empty((0,3))
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    s = confideres
    area = (x2-x1+1) * (y2-y1+1)
    I = np.argsort(s)
    pick = np.zeros_like(s, dtype=np.int16)
    counter = 0
    while len(I)>0:
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
 
        o = inter / (area[i] + area[idx] - inter)
        I = np.where(o<=threshold)
    pick = pick[0:counter]
    return pick
'''
'''
image_path = "5.jpg"
if os.path.exists(image_path):
    img = cv2.imread(image_path)
    if img == None:
        os._exit() 
    
    #img = np.empty((1000, 1000, 3))
confideres = [0.4,0.3,0.5,0.6,0.1]   
lst = [[[100,100],[200,100],[100,200],[200,200]], [[120,120],[210,120],[120,250],[210,250]], [[160,150],[280,150],[160,240],[280,240]], [[80,60],[150,60],[80,170],[150,170]], [[100,150],[150,150],[100,230],[150,230]]]
for i in range(len(lst)):
    tl = lst[i][0]
    tr = lst[i][1]
    bl = lst[i][2]
    br = lst[i][3]
    cv2.rectangle(img, (tl[0],tl[1]), (br[0],br[1]),(0,255,0),1)
cv2.imwrite("test.bmp", img)

boxes = np.array(lst)
res = nms(lst,confideres,0.2)
for i in range(len(res)):
    tl = res[i][0]
    tr = res[i][1]
    bl = res[i][2]
    br = res[i][3]
    cv2.rectangle(img, (tl[0],tl[1]), (br[0],br[1]),(0,0,255),1)
cv2.imwrite("test.bmp", img)
'''
'''        
tl = (100,100)
tr = (200,100)
bl = (100,200)
br = (200,200)

num = 5

labels = []
for i in range(0,num):
    for j in range(i+1,num):
        lst.append(i*j)
        labels.append([i,j])
        print (i,j)
print (lst)
print (labels)

res = []
for i in range(len(lst)):
    if lst[i] < 5:
       res.append(labels[i])
print (res)          


#arr = np.random.randn(4,4)
arr = [2,1,4,13,7,5,40,8,3,20]
print (arr)
print (np.where(arr>5,2,-2))


lst = [2,1,4,13,7,5,40,8,3,20]
I = np.where(lst>3,100,-100)
print ("I:{}\n".format(I))
I = np.argsort(lst)

print ("I:{}\n".format(I))
pick = np.zeros_like(lst, dtype=np.int16)
counter = 0
while I.size>0:
    i = I[-1]
    print ("i:{}\n".format(i))
    pick[counter] = i
    counter += 1
    idx = I[0:-1]
    print ("idx:{}\n".format(idx))   
    dis = np.minimum(lst[i],lst)
    print ("dis:{}\n".format(dis))  
    I = I[np.where(dis>3)]
    pick = pick[0:counter]
    print ("pick:{}\n".format(pick))
#print (ll)
    
s = (3, 2)
z = np.zeros(s, dtype = int)  
zz = np.zeros_like(z) 
#print (zz)
'''

''' tl0 = (100,100)
tr0 = (200,100)
bl0 = (100,200)
br0 = (200,200)

tl1 = (180,100)
tr1 = (250,100)
bl1 = (180,200)
br1 = (250,200)
print (get_twoRect_IOU(tl0,tr0,bl0,br0,tl1,tr1,bl1,br1))
    
  
tl = (100,100)#(100,100)
tr = (200,100)#(200,100)
bl = (50,200)#(100,200)
br = (150,200)#(200,200)
 
pt = (150,150) #center
print (IsPointInMatrix(tl,tr,bl,br,pt))
pt = (50,50)  #tl
print (IsPointInMatrix(tl,tr,bl,br,pt))
pt = (50,250) #bl
print (IsPointInMatrix(tl,tr,bl,br,pt))
pt = (250,50) #tr
print (IsPointInMatrix(tl,tr,bl,br,pt))
pt = (250,250)#br
print (IsPointInMatrix(tl,tr,bl,br,pt))'''
 
'''
def get_rotate_point(tl,tr,bl,br,cos,sin):
    tl = (tl[0]*cos-tl[1]*sin, tl[0]*sin+tl[1]*cos)
    tr = (tr[0]*cos-tr[1]*sin, tr[0]*sin+tr[1]*cos)
    bl = (bl[0]*cos-bl[1]*sin, bl[0]*sin+bl[1]*cos)
    br = (br[0]*cos-br[1]*sin, br[0]*sin+br[1]*cos)
    return (tl,tr,bl,br)
    
image_path = "5.jpg"
if os.path.exists(image_path):
    img = cv2.imread(image_path)
    if img == None:
        os._exit() 
    
    img_rows = img.shape[0]
    img_cols = img.shape[1]
    #img = np.empty((1000, 1000, 3))
    
    ########### src
    tl = (300,100)
    tr = (400,100)
    bl = (300,200)
    br = (400,200)
    center0 = ((tl[0]+tr[0])/2, (tl[1]+bl[1])/2)
    
    cv2.circle (img,(center0[0],center0[1]),2,(0,0,0),1)
    cv2.line(img,(int(tl[0]),int(tl[1])),(int(tr[0]),int(tr[1])),(0,0,255),3)
    cv2.line(img,(int(tl[0]),int(tl[1])),(int(bl[0]),int(bl[1])),(0,0,255),3)
    cv2.line(img,(int(bl[0]),int(bl[1])),(int(br[0]),int(br[1])),(0,0,255),3)
    cv2.line(img,(int(tr[0]),int(tr[1])),(int(br[0]),int(br[1])),(0,0,255),3)
    
    cos = 0.5
    sin = 1.732/2 
    
    ###### rorate
    (tl,tr,bl,br) = get_rotate_point(tl,tr,bl,br,cos,sin)  
    
    x_min = min(tl[0],tr[0],bl[0],br[0])
    x_max = max(tl[0],tr[0],bl[0],br[0])
    y_min = min(tl[1],tr[1],bl[1],br[1])
    y_max = max(tl[1],tr[1],bl[1],br[1])
    center1 = (float((x_min + x_max)) / 2.0, float((y_min + y_max) / 2.0))
    
    cv2.circle (img,(int(center1[0]),int(center1[1])),2,(0,0,0),1)
    cv2.line(img,(int(tl[0]),int(tl[1])),(int(tr[0]),int(tr[1])),(0,255,255),1)
    cv2.line(img,(int(tl[0]),int(tl[1])),(int(bl[0]),int(bl[1])),(0,255,255),1)
    cv2.line(img,(int(bl[0]),int(bl[1])),(int(br[0]),int(br[1])),(0,255,255),1)
    cv2.line(img,(int(tr[0]),int(tr[1])),(int(br[0]),int(br[1])),(0,255,255),1)
    
    ########
    cos = 0.5
    sin = -1.732/2
    
    (tl,tr,bl,br) = get_rotate_point(tl,tr,bl,br,cos,sin)   
    cv2.line(img,(int(tl[0]),int(tl[1])),(int(tr[0]),int(tr[1])),(255,0,0),1)
    cv2.line(img,(int(tl[0]),int(tl[1])),(int(bl[0]),int(bl[1])),(255,0,0),1)
    cv2.line(img,(int(bl[0]),int(bl[1])),(int(br[0]),int(br[1])),(255,0,0),1)
    cv2.line(img,(int(tr[0]),int(tr[1])),(int(br[0]),int(br[1])),(255,0,0),1)
    print (tl,tr,bl,br)
    
    ######### move
    dx = center1[0] - center0[0]
    dy = center1[1] - center0[1]
      
    tl = (tl[0]-dx, tl[1]-dy)
    tr = (tr[0]-dx, tr[1]-dy)
    bl = (bl[0]-dx, bl[1]-dy)
    br = (br[0]-dx, br[1]-dy)
    
    cv2.line(img,(int(tl[0]),int(tl[1])),(int(tr[0]),int(tr[1])),(255,0,255),1)
    cv2.line(img,(int(tl[0]),int(tl[1])),(int(bl[0]),int(bl[1])),(255,0,255),1)
    cv2.line(img,(int(bl[0]),int(bl[1])),(int(br[0]),int(br[1])),(255,0,255),1)
    cv2.line(img,(int(tr[0]),int(tr[1])),(int(br[0]),int(br[1])),(255,0,255),1)
        
    cv2.imwrite("test.bmp", img)'''