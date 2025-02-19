import pickle
import sys,os
import numpy as np
from softmax import softmax
from 神经网络 import sigmoid
sys.path.append(os.pardir)
from mnist2 import load_mnist
from PIL import Image

def mean_squared_error(y,t):
    return 0.5*np.sum((y-t)**2)

def cross_entropy_error(y,t):
    if y.ndim==1:
        t=t.reshape(1,t.size)
        y=y.reshape(1,y.size)
    return -np.sum(t*np.log(y+1e-7))/batch_size

def numercial_diff(f,x):
    h=1e-4
    return (f(x+h)-f(x-h))/2*h

(x_train,t_train),(x_test,t_test)=load_mnist(normalize=True,one_hot_label=True)
train_size=x_train.shape[0]
batch_size=10
batch_mask=np.random.choice(train_size,batch_size)
x_batch_mask=x_train[batch_mask]
t_batch_mask=t_train[batch_mask]

