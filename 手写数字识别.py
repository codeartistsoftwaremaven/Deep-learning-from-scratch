import pickle
import sys,os
import numpy as np

from softmax import softmax
from 神经网络 import sigmoid

sys.path.append(os.pardir)
from mnist2 import load_mnist
from PIL import Image

def img_show(img):
    pil_img=Image.fromarray(np.uint8(img))
    pil_img.show()

def get_data():
    (x_train,t_train),(x_test,t_test)=load_mnist(flatten=True,normalize=False,one_hot_label=False)
    return x_test,t_test

def init_network():
    with open(r"C:\Users\86151\Desktop\【源代码】深度学习入门：基于Python的理论与实现_20240716\ch03\sample_weight.pkl",'rb') as f:
        network=pickle.load(f)

    return network

def predict(network,x):
    W1,W2,W3=network['W1'],network['W2'],network['W3']
    b1,b2,b3=network['b1'],network['b2'],network['b3']

    a1=np.dot(x,W1)+b1
    z1=sigmoid(a1)
    a2=np.dot(z1,W2)+b2
    z2=sigmoid(a2)
    a3=np.dot(z2,W3)+b3
    y=softmax(a3)

    return y

x,t=get_data()
batch_size=100
network=init_network()

accuracy_cut=0
for i in range(0,len(x),batch_size):
    x_batch=x[i:i+batch_size]
    y_batch=predict(network, x[i:i+batch_size])
    p=np.argmax(y_batch,axis=1)
    accuracy_cut+=np.sum(p==t[i:i+batch_size])

print('accuracy:' + str(float(accuracy_cut)/len(x)))
