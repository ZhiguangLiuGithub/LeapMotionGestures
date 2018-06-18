import os
import numpy as np
import sys

################################################
#########       GLOBAL VARIABLES      ##########
################################################
location = ""
my_data_location = "../visualize/"
finger_names = ["TYPE_THUMB", "TYPE_INDEX", "TYPE_MIDDLE", "PALM", "TYPE_RING", "TYPE_PINKY"]
minimum = [150,100,0]
ranges  = [750,500,600]
img_rows = 50
img_cols = 50
NO_INFO = False
#max_v = np.array([[15.29, 20.92, 16.35], [ 8.99, 16.15, 11.92], [12.13, 19.84, 15.06], [18.23, 20.62, 21.93], [12.02, 25.24, 15.59], [13.15, 18.8 , 17.37]])

#max_v = np.array([[18.09092683, 28.46656277, 28.67797005], [16.86947857, 28.18406665, 22.46399137], [24.11379573, 39.75088479, 31.89060003],
#          [21.60760976, 42.81772681, 46.89036457], [20.56626923, 29.64425473, 42.53967063], [25.01218211, 30.00609044, 43.22103779]])

max_v = np.array([[18.09092683, 28.46656277, 28.67797005],
         [16.86947857, 28.18406665, 22.46399137],
         [24.11379573, 39.75088479, 31.89060003],
         [21.60760976, 42.81772681, 46.89036457],
         [20.56626923, 29.64425473, 42.53967063],
         [25.01218211, 30.00609044, 43.22103779]])

min_v = max_v*0

################################################
#########        PREPROCESSING       ###########
################################################
from scipy import ndimage
import scipy
import numpy as np
from skimage import transform

#Preprocessing Steps:
#1. create a 200X200 image in XY plane
def createPlanes(data, invertY = False, rowsX = 200, rowsY = 200, rowsZ = 200):
    idxX = rowsX * (data[::3] - minimum[0]) / ranges[0] 
    if invertY:
        idxY = rowsY * (-1*(data[1::3] - minimum[1]) + ranges[1] - minimum[1]) / ranges[1]
    else:
        idxY = rowsY * (data[1::3] - minimum[1]) / ranges[1] 
    idxZ = rowsZ * (data[2::3] - minimum[2]) / ranges[2] 

    matYX = np.zeros((rowsY,rowsX))
    matYZ = np.zeros((rowsY,rowsZ))
    matZX = np.zeros((rowsZ,rowsX))

    timeYX = np.zeros((rowsY,rowsX))
    timeYZ = np.zeros((rowsY,rowsZ))
    timeZX = np.zeros((rowsZ,rowsX))

    for i in range(0,len(idxX)):
        r = min(int(round(idxY[i])), rowsY - 1) if idxY[i] > 0 else 0
        c = min(int(round(idxX[i])), rowsX - 1) if idxX[i] > 0 else 0
        
        x = min(int(round(idxX[i])), rowsX - 1) if idxX[i] > 0 else 0
        y = min(int(round(idxY[i])), rowsY - 1) if idxY[i] > 0 else 0
        z = min(int(round(idxZ[i])), rowsZ - 1) if idxZ[i] > 0 else 0
        
        #comment to not consider order
        timeYX[y][x] = i+1 - timeYX[y][x]
        timeYZ[y][z] = i+1 - timeYZ[y][z]
        timeZX[z][x] = i+1 - timeZX[z][x]
        #not use 1 but Vel or Dir
        matYX[y][x] = matYX[y][x] + 1*timeYX[y][x]
        matYZ[y][z] = matYZ[y][z] + 1*timeYZ[y][z]
        matZX[z][x] = matZX[z][x] + 1*timeZX[z][x]
    #Gray value depends on relative time (non absolute)
    return matYX/len(idxX), matYZ/len(idxX), matZX/len(idxX)  
    #return matYX, matYZ, matZX

#2. Apply dilation
def dilation(mat, dim = 5):
    kernel = np.ones((5,5))
    #ndimage.binary_dilation(mat, structure=kernel).astype(mat.dtype)
    return ndimage.grey_dilation(mat, size=(5,5))

    
#3. Crop
def crop(mat, margin = 5):
    m = np.nonzero(mat)
    max_r = np.max(m[0])
    max_c = np.max(m[1])
    min_r = np.min(m[0])
    min_c = np.min(m[1])
    return mat[max(min_r - margin, 0): min(max_r + 1 + margin, mat.shape[0]), max(min_c - margin, 0): min(max_c + 1 + margin, mat.shape[1])]
#4. resize image
def resize(mat, dim = (50,50)):
    #return scipy.misc.imresize(mat, dim, mode = 'F')
    return transform.resize(mat, dim, preserve_range = True)

# Do all preprocess steps
def preprocess(data, invertY = False):
    imageYX, imageYZ, imageZX = createPlanes(data, invertY)
    #Dilation
    imageYX = dilation(imageYX)
    imageYZ = dilation(imageYZ)
    imageZX = dilation(imageZX)
    #Crop
    #imageYX = crop(imageYX)
    #imageYZ = crop(imageYZ)
    #imageZX = crop(imageZX)
    #Resize
    imageYX = resize(imageYX)
    imageYZ = resize(imageYZ)
    imageZX = resize(imageZX)
    return [imageYX, imageYZ, imageZX]

#One Hot Encode
from numpy import array
from numpy import argmax

data_classes = ["capE", "CheckMark", "e", "F", "Figure8", "Tap", "Tap2", "Grab", "Pinch", "Release", "Swipe", "Wipe"]
int_to_label = {}
label_to_int = {}

for i,c in enumerate(data_classes):
    int_to_label[i] = c
    label_to_int[c] = i

def encode(labels):
    m = np.zeros((len(labels), len(data_classes)))
    for i in range(0, len(labels)):
        m[i][label_to_int[labels[i]]] = 1
    return m

def decode(values):
    m = [" "] * len(values)
    for i in range(0, len(m)):
        m[i] = int_to_label[np.argmax(values[i])]
    return np.array(m)

################################################
#########         LOAD DATA           ##########
################################################
def getPosition(line, finger = "TYPE_INDEX", pos = None):
    line_ = line.split(" ")
    if(pos != None):
        if "NP" in line_[pos:pos+3]:
            return []
        r = [float(i) for i in line_[pos:pos+3]]
        return r
    idx = [i for i,s in enumerate(line_) if s == finger]
    if(len(idx) == 0):
        return []
    idx = idx[0] + 5
    r = [float(i) for i in line_[idx:idx+3]]
    if(len(r) != 3):
        print("ERROR!")
        return (None)
    return r


def load_data(location, name):
    my_fingers = {}
    file  = open(location + name, "r") 
    lines = file.readlines()
    for finger in ["PALM","TYPE_THUMB", "TYPE_INDEX", "TYPE_MIDDLE", "TYPE_RING", "TYPE_PINKY"]:
        data = []
        features = []
        pos = None
        if(finger == "PALM"):
            print(lines[1])	
            for k,s in enumerate(lines[1].split(" ")):
                if s == "h_PosX":
                    pos = k
        for line in lines[2:]:
            f = getPosition(line, finger, pos) 
            if len(f) > 0:
                features = features + f
        if len(features) > 0:
            data.append(np.array(features))
        file.close()
        sys.stdout.write('\r')
        sys.stdout.write("Finger: %s %s " % (location + name, finger))
        sys.stdout.flush()                
        my_fingers[finger] = data
        if len(data) == 0:
            NO_INFO = True

    return my_fingers



def get_images(my_fingers):
    my_data = {}
    print("Preprocessing")
    for finger in my_fingers:
        my_data[finger] = []
        for i,gesture in enumerate(my_fingers[finger]):
            image = preprocess(gesture, True)
            my_data[finger].append(image)
            sys.stdout.write("Image: %d" % (i))
            sys.stdout.write('\r')
            sys.stdout.flush()
    return my_data


def merge(my_data):
    my_images = np.zeros((len(my_data["TYPE_INDEX"]),6,3,50,50))
    for i in range(0, len(my_data["TYPE_INDEX"])):
        sys.stdout.write('\r')
        sys.stdout.write("Image: %d" % (i))
        sys.stdout.flush()
        image = np.zeros((6,3,50,50))
        for j,finger in enumerate(['TYPE_THUMB', 'TYPE_INDEX', 'TYPE_MIDDLE', 'PALM', 'TYPE_RING', 'TYPE_PINKY']):
            image[j][0] = resize(crop(my_data[finger][i][0]))
            image[j][1] = resize(crop(my_data[finger][i][1]))
            image[j][2] = resize(crop(my_data[finger][i][2]))
        my_images[i] = image
    return my_images


def normalize(mat, max_v, min_v, scale = 255):
    ids = np.nonzero(mat)
    mat[ids] = scale * ( mat[ids] - min_v ) / (max_v - min_v)
    m = mat
    return mat

def normalize_data(my_images):
    for i,image in enumerate(my_images):
        for j,finger in enumerate(image):
            info = True
            for k,channel in enumerate(finger):
                channel_n = normalize(channel, max_v[j][k], min_v[j][k])
                if channel_n == []:
                    print("image: " + str(i) + " contains no information in channel " + str(k))
                    info = False

def channel_last(data):
    data_ = np.zeros((data.shape[0], data.shape[1], data.shape[3], data.shape[4], data.shape[2]))
    for i,sample in enumerate(data):
        for j,finger in enumerate(sample):
            rgbArray = np.zeros((finger.shape[1],finger.shape[2],3))
            rgbArray[:,:, 0] = finger[0]
            rgbArray[:,:, 1] = finger[1]
            rgbArray[:,:, 2] = finger[2]
            data_[i][j] = rgbArray 
    return data_

def scale_data(my_images):
    #scale
    data_im = np.zeros((my_images.shape[0], my_images.shape[1], img_rows, img_cols, 3))
    new_shape = (img_rows, img_cols, 3)
    print("SCALE")
    for i,sample in enumerate(my_images):
        for j,image in enumerate(sample):
            im = transform.resize(image, new_shape)
            im = image/255.0
            data_im[i][j] = im
            sys.stdout.write('\r')
            sys.stdout.write("Image: %d" % (i))
            sys.stdout.flush()       
    return data_im

def swap(data):
    return np.swapaxes(data,1,4)

def predict_data(model_alt_3, data_im):
    data_im_t = swap(data_im)
    my_d_t = []
    for i in range(0,3):
        my_d_t.append(data_im_t[:,i])
    predicted = decode(model_alt_3.predict(my_d_t, verbose=1))
    return predicted


#Create same model for each channel
#create a small model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Merge
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential, Model
from keras import optimizers
import keras
import numpy as np

def create_model():
    dirs = ["capE", "CheckMark", "e", "F", "Figure8", "Tap", "Tap2", "Grab", "Pinch", "Release", "Swipe", "Wipe"]

    input_shape = (img_rows, img_cols, 6)

    models_alt = []
    for i in range(0,3):
        model_alt_3d = Sequential()
        model_alt_3d.add(Conv2D(50, kernel_size=(5, 5),
                         activation='relu', input_shape=input_shape))
        model_alt_3d.add(MaxPooling2D(pool_size=(2, 2)))
        model_alt_3d.add(Conv2D(50, (5, 5), activation='relu'))
        model_alt_3d.add(MaxPooling2D(pool_size=(2, 2)))
        model_alt_3d.add(Conv2D(64, (3, 3), activation='relu'))
        model_alt_3d.add(Dropout(0.25))
        model_alt_3d.add(Flatten())
        models_alt.append(model_alt_3d)

    #concatenate
    model_alt_3 = Sequential()
    model_alt_3.add(Merge(models_alt, mode = 'concat'))
    # dense layers
    model_alt_3.add(Dense(128, activation='relu'))
    model_alt_3.add(Dropout(0.5))
    model_alt_3.add(Dense(len(dirs), activation='softmax'))

    model_alt_3.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    print(model_alt_3.summary())
    return model_alt_3


def setup_model():
    model_alt_3 = create_model()
    model_alt_3.load_weights(location + 'model.h5')
    return model_alt_3

def execute(model_alt_3, name):
    NO_INFO = False
    my_fingers = load_data(my_data_location, name)
    if NO_INFO == True:
        return "NO INFORMATION"
    my_data = get_images(my_fingers)
    my_images = merge(my_data)
    normalize_data(my_images)
    my_images = channel_last(my_images)
    data_im = scale_data(my_images)
    return predict_data(model_alt_3, data_im)


model_alt_3 = setup_model()

import socket
print('Start Connection')

HOST = ''                 
PORT = 50007              
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

s.bind((HOST, PORT))
s.listen(1)

conn, addr = s.accept()

print('Connected by', addr)
while True:
    data = conn.recv(1024)
    if not data: break

    if "gesture" in data.decode("utf-8"):     
        print(data) 
        response = execute(model_alt_3, data.decode("utf-8"))
        print("Predicted : " + str(response))
        if len(response) > 0:
            conn.send(response[0].encode("utf-8"))

    
conn.close()




