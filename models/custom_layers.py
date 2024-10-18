
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K


class T2V(Layer):
    
    def __init__(self, output_dim=None, **kwargs):
        self.output_dim = output_dim
        super(T2V, self).__init__(**kwargs)
        
    def build(self, input_shape):       
        self.W = self.add_weight(name='W', shape=(input_shape[-1], self.output_dim), initializer='uniform', trainable=True)        
        self.P = self.add_weight(name='P', shape=(input_shape[1], self.output_dim), initializer='uniform', trainable=True)        
        self.w = self.add_weight(name='w', shape=(input_shape[1], 1), initializer='uniform', trainable=True)        
        self.p = self.add_weight(name='p', shape=(input_shape[1], 1), initializer='uniform', trainable=True)        
        super(T2V, self).build(input_shape)
        
    def call(self, x):
        
        original = self.w * x + self.p
        sin_trans = K.sin(K.dot(x, self.W) + self.P)
        
        return K.concatenate([sin_trans, original], -1)

class CustomAttention(Layer):
    def __init__(self, **kwargs):
        super(CustomAttention,self).__init__(**kwargs)
 
    def build(self,input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1), 
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1), 
                               initializer='zeros', trainable=True)        
        super(CustomAttention, self).build(input_shape)
 
    def call(self,x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x,self.W)+self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)   
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = K.sum(context, axis=1) # Todo: is is better with or without the sum?
        return context
