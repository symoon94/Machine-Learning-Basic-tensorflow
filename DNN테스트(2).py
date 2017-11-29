#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 16:28:01 2017

@author: moonsooyoung
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 16:20:48 2017

@author: moonsooyoung
"""
This is an H1
===============
import os
import numpy as np

import tensorflow as tf
import matplotlib.pylab as plt
import PIL.Image as pilimg

#DeepNeuralNetwork에 쓸 각 layer에 사용할 노드의 갯수를 사용자가 원하는 만큼 정해줌.
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

#n_classes = 데이터의 클래스 수.
n_classes = 10

#input과 output의 placeholder을 만들어줌. [None, 784]는 input data의 사이즈.
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

#신경망 구조에 들어간 변수
w1 = tf.Variable(tf.random_normal([784, n_nodes_hl1]))
b1 = tf.Variable(tf.random_normal([n_nodes_hl1]))

w2 = tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2]))
b2 = tf.Variable(tf.random_normal([n_nodes_hl2]))

w3 = tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3]))
b3 = tf.Variable(tf.random_normal([n_nodes_hl3]))

w4 = tf.Variable(tf.random_normal([n_nodes_hl3, n_classes]))
b4 = tf.Variable(tf.random_normal([n_classes]))

#변수들 저장된 파일경로
save_path = 'machine/'
model_name = 'DNN_md'
if not os.path.exists(save_path):
    os.makedirs(save_path)

save_path_full = os.path.join(save_path, model_name)

#테스트 단계
#테스트할 파일을 불러옴.
k = pilimg.open('/Users/moonsooyoung/Desktop/수영python과제/seven.png' )
plt.imshow(k)
imgarray=np.array(k)    #컬러채널이 1개가 되도록 이미 전처리를 한 상태라서 그냥 (28,28)로 나옴.
kkk = imgarray/255    #k1 벡터 안의 숫자들을 0과 1 사이로 normalize시키기 벡터 내의 가장 큰 값으로 k1을 나눠줌.
sydata=kkk.reshape(1,784)    #훈련할 때와 같은 input data 형태로 맞춰줌. (28,28)->(1,784).
x_data = tf.cast(sydata, 'float')

#변수 불러오기
sess=tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver([w1, w2, w3, w4, b1, b2, b3, b4])    #전에 저장해둔 변수를 불러옴.
new_saver = tf.train.import_meta_graph('/Volumes/SM/machine/DNN_md.meta')    
#학습 단계에서 자동으로 생긴 meta file에 변수들의 값이 저장되어 있음. saver.restore()로 그 값들을 불러옴.
saver.restore(sess,save_path_full)

#처음에 썼던 신경망 알고리즘을 그대로 써주고, input data넣는 x자리에 테스트 할 데이터(x_data)를 바꿔 써줌.
l1 = tf.add(tf.matmul(x_data,w1), b1)
l1 = tf.nn.relu(l1)

l2 = tf.add(tf.matmul(l1,w2), b2)
l2 = tf.nn.relu(l2)

l3 = tf.add(tf.matmul(l2,w3), b3)
l3 = tf.nn.relu(l3)

output = tf.matmul(l3,w4) + b4

#테스트 시작
print('============================================TEST 결과============================================')
print(sess.run (output))
print(sess.run(tf.argmax(output, 1)))
