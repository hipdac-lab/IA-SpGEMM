from skimage import io,transform
import numpy as np
from keras.models import *
from keras.layers import *
import keras
import os
from keras import backend as K
np.set_printoptions(threshold=np.inf)


def read_data(dir_str):
    data_temp=[]
    with open(dir_str) as fdata:
        while True:
            line=fdata.readline()
            if not line:
                break
            data_temp.append([float(i) for i in line.split()])
    return np.array(data_temp)



def Pred(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r):

    feature_input = [a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r]
    print("matrix features:", feature_input)
    
    img_1 = read_data("imgs/img1.txt")
    max_num = np.max(img_1)
    img_1 = np.reshape(img_1, (128, 128));
    img_1 = img_1 * 255 / max_num;

    img_2 = read_data("imgs/img2.txt")
    max_num = np.max(img_2)
    img_2 = np.reshape(img_2, (128, 128));
    img_2 = img_2 * 255 / max_num;


    Test_img_1 = np.reshape(img_1,[1,128,128,1])
    Test_img_2 = np.reshape(img_2,[1,128,128,1])
    Test_feature = np.reshape(feature_input,[1,18])

    input1 = keras.layers.Input(shape=(128,128,1))
    conv1_1 = keras.layers.Conv2D(16,(3,3), activation='tanh')(input1)
    max_pooling1_1 = keras.layers.MaxPooling2D(2,2)(conv1_1)
    conv1_2 = keras.layers.Conv2D(16,(5,5), strides=(2, 2), padding='same',  activation='tanh')(max_pooling1_1)
    max_pooling1_2 = keras.layers.MaxPooling2D(2,2)(conv1_2)

    conv1_3 = keras.layers.Conv2D(16,(5,5), strides=(2, 2), padding='same',  activation='tanh')(max_pooling1_2)
    max_pooling1_3 = keras.layers.MaxPooling2D(2,2)(conv1_3)

    flatten1 = keras.layers.Flatten()(max_pooling1_3)


    input2 = keras.layers.Input(shape=(128,128,1))
    conv2_1 = keras.layers.Conv2D(16,(3,3), activation='tanh')(input2)
    max_pooling2_1 = keras.layers.MaxPooling2D(2,2)(conv2_1)

    conv2_2 = keras.layers.Conv2D(16,(5,5), strides=(2, 2), padding='same',  activation='tanh')(max_pooling2_1)
    max_pooling2_2 = keras.layers.MaxPooling2D(2,2)(conv2_2)

    conv2_3 = keras.layers.Conv2D(16,(5,5), strides=(2, 2), padding='same',  activation='tanh')(max_pooling2_2)
    max_pooling2_3 = keras.layers.MaxPooling2D(2,2)(conv2_3)

    flatten2 = keras.layers.Flatten()(max_pooling2_3)

    input3 = keras.layers.Input(shape=(18,))
    feature_dense1 = keras.layers.Dense(18, activation='tanh')(input3)

    image_dense1 = keras.layers.Dense(32, activation='tanh')(flatten1)
    image_dense2 = keras.layers.Dense(32, activation='tanh')(flatten2)
    added_layer = keras.layers.Concatenate()([image_dense1, image_dense2, feature_dense1])

    output= keras.layers.Dense(3, activation='softmax')(added_layer)

    model = keras.models.Model(inputs=[input1,input2,input3], outputs=output)

    model.load_weights('./NetWeights/P100_weights.h5')

    #model.summary()

    result = model.predict([Test_img_1, Test_img_2, Test_feature])

    Chosen_One = np.argmax(result[0])

    #print(Chosen_One)

    K.clear_session()

    return Chosen_One


