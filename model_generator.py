import math
import numpy as np 
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.contrib.layers.python.layers import batch_norm


def batchnorm(inputT, is_training=False, scope=None):
#    return inputT
    
    # Note: is_training is tf.placeholder(tf.bool) type
    is_training = tf.get_collection('istrainvar')[0]
    return tf.cond(is_training,
                   lambda: batch_norm(inputT, is_training=True,
                                      center=True, scale=True, decay=0.9, updates_collections=None, scope=scope),
                   lambda: batch_norm(inputT, is_training=False,
                                      center=True, scale=True, decay=0.9, updates_collections=None, scope=scope,
                                      reuse=True))
 
def stability_loss(inp, sample_size):
    
#    if randvec is not None:
#        o1 = tf.gather(inp,randvec[0:sample_size])
#        me1, var1 = tf.nn.moments(o1,0)
#        o2 = tf.gather(inp,randvec[sample_size:2*sample_size])
#        me2, var2 = tf.nn.moments(o2,0)
#        shape = var1.get_shape()
##        eps1 = tf.get_variable("var1", (shape[0]), tf.float32, tf.constant_initializer(0.1))
#    else:    
        me1, var1 = tf.nn.moments(inp[0:sample_size,:],0)
        me2, var2 = tf.nn.moments(inp[sample_size:2*sample_size,:],0)
        shape = var1.get_shape()
        eps1 = tf.get_variable("var2", (shape[0]), tf.float32, tf.constant_initializer(0.1))
        var1 = tf.abs(var1) 
        var2 = tf.abs(var2) 
        tf.add_to_collection('l2_norm',(tf.reduce_mean(tf.square(1 - (var1)/(var2+eps1)))))
    
    
                  
def linear(input_, output_size, sample_size, eps, scope=None, bn = False, activation = None, hidden = True):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable(scope + "Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=1/np.sqrt(shape[1])))
#        we = tf.reshape(matrix,(shape[1]*output_size,1))
#        randvec = tf.cast(tf.random_uniform((2*sample_size,1),0,shape[1]*output_size),tf.int32)
#        stability_loss(we,sample_size,eps,randvec)

        tf.add_to_collection('l2_loss',(tf.nn.l2_loss(matrix)))
        bias = tf.get_variable(scope + "bias", [output_size],
            initializer=tf.constant_initializer(0.0))
        output = tf.matmul(input_, matrix) + bias                
        if bn:
            output = batchnorm(output, scope = scope)
        if hidden:
            stability_loss(output,sample_size)
        if activation:
            output = activation(output)
#        if hidden:
#            stability_loss(output,sample_size,eps)
#        if activation:
#            output = activation(output)
        return output




def conic_architecture(inp, init_dim, num_of_labels, activation, depth, sample_size, eps, isTrain_node, keep = 1, bn_ = 0, conic = 0):
    bn = bn_==1
    factor = (np.power(float(num_of_labels)/float(init_dim),1/float(depth)))
    tmp = linear(inp, init_dim, sample_size, eps, scope='init', bn = bn, activation = activation, hidden = False)
#    tmp = tf.cond(isTrain_node, lambda: tf.nn.dropout(tmp, keep), lambda: tmp)
    dim = init_dim
    for i in range(depth - 1):
        if conic==1:
            dim = (dim*factor).astype('int')
        tmp = linear(tmp, dim, sample_size, eps, scope='layer_'+str(i), bn = bn, activation = activation)
#        tmp = tf.cond(isTrain_node, lambda: tf.nn.dropout(tmp, keep), lambda: tmp)
    tmp = tf.cond(isTrain_node, lambda: tf.nn.dropout(tmp, keep), lambda: tmp)
    out = linear(tmp, num_of_labels, sample_size, eps, scope='out', bn = False, hidden = False)   
    return out

def conic_architecture_selu(inp, init_dim, num_of_labels, activation, depth, sample_size, eps, isTrain_node, keep = 1, bn_ = 0, conic = 0):
    bn = bn_==1
    factor = (np.power(float(num_of_labels)/float(init_dim),1/float(depth)))
    tmp = linear(inp, init_dim, sample_size, eps, scope='init', bn = bn, activation = activation, hidden = False)
    tmp = tf.cond(isTrain_node, lambda: tf.contrib.nn.alpha_dropout(tmp, keep), lambda: tmp)
    dim = init_dim
    for i in range(depth - 1):
        if conic==1:
            dim = (dim*factor).astype('int')
        tmp = linear(tmp, dim, sample_size, eps, scope='layer_'+str(i), bn = bn, activation = activation)
        tmp = tf.cond(isTrain_node, lambda: tf.contrib.nn.alpha_dropout(tmp, keep), lambda: tmp)
    out = linear(tmp, num_of_labels, sample_size, eps, scope='out', bn = False, hidden = False)   
    return out
     
        
        