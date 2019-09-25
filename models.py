# models.py

from keras import Model 
from keras.layers import Embedding, Lambda, Multiply, Reshape, Activation, Dropout, Concatenate, Flatten, Conv2D, Dense, BatchNormalization
import tensorflow as tf

from keras import backend as K

class TransE(Model):
    def __init__(self, ndim, mdim, kdim = 128, embeddings_regularizer=None, embeddings_constraint=None, dropout=0.2):
        super(TransE, self).__init__(name='transe')
        self.e = Embedding(ndim, kdim, embeddings_regularizer=embeddings_regularizer,embeddings_constraint=embeddings_constraint)
        
        self.r = Embedding(mdim, kdim)
        
        self.l = Lambda(lambda x:tf.norm(x,axis=1))
        self.reshape = Reshape((-1,))
        self.bn = BatchNormalization()
        self.d = Dropout(dropout)
        
    def call(self, inputs):
        s,p,o = inputs[:,0],inputs[:,1],inputs[:,2]
        x = self.e(s) + self.r(p) - self.e(o)
        x = self.d(x)
        x = self.l(x)
        x = Reshape((-1,1))(x)
        x = self.bn(x)
        return self.reshape(x)
        

class DistMult(Model):
    def __init__(self, ndim, mdim, kdim = 128, embeddings_regularizer=None,embeddings_constraint=None, dropout=0.2):
        super(DistMult, self).__init__(name='distmult')
        self.e = Embedding(ndim, kdim, embeddings_regularizer=embeddings_regularizer,embeddings_constraint=embeddings_constraint)
        
        self.r = Embedding(mdim, kdim)
        
        self.d = Dropout(dropout)
        self.l = Multiply()
        self.s = Lambda(lambda x: K.sum(x, axis=-1))
        self.a = Activation('sigmoid')
        self.reshape = Reshape((-1,))
        self.bn = BatchNormalization()
    
    def call(self, inputs):
        s,p,o = inputs[:,0],inputs[:,1],inputs[:,2]
        x = [self.d(self.e(s)),self.d(self.r(p)),self.d(self.e(o))]
        x = self.l(x)
        x = self.s(x)
        x = self.a(x)
        x = Reshape((-1,1))(x)
        x = self.bn(x)
        x = self.reshape(x)
        return x
    
def circular_cross_correlation(x, y):
    """Periodic correlation, implemented using the FFT.
    x and y must be of the same length.
    """
    return tf.real(tf.signal.ifft(tf.multiply(tf.math.conj(tf.signal.fft(tf.cast(x, tf.complex64))) , tf.signal.fft(tf.cast(y, tf.complex64)))))

def HolE_fn(s,p,o):
    # sigm(p^T (s \star o))
    # dot product in tf: sum(multiply(a, b) axis = 1)
    return tf.reduce_sum(tf.multiply(p, circular_cross_correlation(s, o)), axis = 1)
        
class HolE(Model):
    def __init__(self, ndim, mdim, kdim = 128, embeddings_regularizer=None,embeddings_constraint=None, dropout=0.2):
        super(HolE, self).__init__(name='hole')
    
        self.e = Embedding(ndim, kdim, embeddings_regularizer=embeddings_regularizer,embeddings_constraint=embeddings_constraint)
        
        self.d = Dropout(dropout)
        self.r = Embedding(mdim, kdim)
        
        self.l = Lambda(lambda x: HolE_fn(x[0],x[1],x[2]))
        self.a = Activation('sigmoid')
        self.reshape = Reshape((-1,))
        self.bn = BatchNormalization()
        
    def call(self, inputs):
        s,p,o = inputs[:,0],inputs[:,1],inputs[:,2]
        x = [self.d(self.e(s)),self.d(self.r(p)),self.d(self.e(o))]
        x = self.l(x)
        x = self.a(x)
        x = Reshape((-1,1))(x)
        x = self.bn(x)
        x = self.reshape(x)
        return x


def ComplEx_fn(real_s, real_o, real_p, img_s, img_o, img_p):
    return tf.reduce_sum(real_s * real_o * real_p, axis = 1) + tf.reduce_sum(img_s * img_o * real_p, axis = 1) + tf.reduce_sum(real_s * img_o * img_p, axis = 1) - tf.reduce_sum(img_s * real_o * img_p, axis = 1)


class ComplEx(Model):
    def __init__(self, ndim, mdim, kdim = 128, embeddings_regularizer=None,embeddings_constraint=None, dropout=0.2):
        super(ComplEx, self).__init__(name='complex')
        
        self.e = Embedding(ndim, kdim, embeddings_regularizer=embeddings_regularizer,embeddings_constraint=embeddings_constraint)
        
        self.r = Embedding(mdim, kdim)
        
        self.d = Dropout(dropout)
        self.e_complex = Embedding(ndim, kdim, embeddings_regularizer=embeddings_regularizer,embeddings_constraint=embeddings_constraint)
        
        self.r_complex = Embedding(mdim, kdim)
        
        self.l = Lambda(lambda x: ComplEx_fn(x[0],x[1],x[2],x[3],x[4],x[5]))
        self.a = Activation('sigmoid')
        self.reshape = Reshape((-1,))
        self.bn = BatchNormalization()
        
    
    def call(self, inputs):
        s,p,o = inputs[:,0],inputs[:,1],inputs[:,2]
        x = [self.e(s),self.r(p),self.e(o),self.e_complex(s),self.r_complex(p),self.e_complex(o)]
        x = [self.d(a) for a in x]
        x = self.l(x)
        x = self.a(x)
        x = Reshape((-1,1))(x)
        x = self.bn(x)
        x = self.reshape(x)
        return x
    
    
class ConvE(Model):
    def __init__(self, ndim, mdim, kdim = 128, width = 16, height = 8, embeddings_regularizer=None,embeddings_constraint=None, dropout=0.2):
        super(ConvE, self).__init__(name='conve')
        
        self.e = Embedding(ndim, kdim, embeddings_regularizer=embeddings_regularizer,embeddings_constraint=embeddings_constraint)
        
        self.r = Embedding(mdim, kdim)
        
        self.reshape = Reshape((width, height, 1))
        self.concat = Concatenate(axis=-2)
        self.d1 = Dropout(dropout)
        self.bn1 = BatchNormalization()
        self.d2 = Dropout(dropout)
        self.bn2 = BatchNormalization()
        
        self.conv = Conv2D(32, (3,3), activation='relu')
        self.f = Flatten()
        self.dense = Dense(kdim, activation='relu')
        self.m = Multiply()
        self.s = Lambda(lambda x: K.sum(x, axis=-1))
        self.ac = Activation('sigmoid')
        self.reshape2 = Reshape((-1,))
        
    def call(self,inputs):
        s,p,o = inputs[:,0],inputs[:,1],inputs[:,2]
        s,p,o = self.e(s), self.r(p), self.e(o)
        
        s = self.reshape(s)
        p = self.reshape(p)
        
        x = self.concat([s,p])
        x = self.d1(x)
        x = self.conv(x)
        x = self.bn1(x)
        x = self.f(x)
        x = self.dense(x)
        x = self.d2(x)
        x = self.bn2(x)
        
        x = self.m([x,o])
        x = self.s(x)
        x = self.ac(x)
        x = self.reshape2(x)
        
        return x
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
