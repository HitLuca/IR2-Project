




def maxpool(x, k, stride):
    '''
    MaxPooling 
    '''
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, stride, stride, 1],
                          padding='SAME')

def inner_product(x, weights, biases, relu=False, name=""):
    '''
    Applies a inner product of the input with the weights. 
    The added biasses and applied relu make this a fully conneted
    layer. 
    '''
    with tf.variable_scope(name):
        # Create variable named "weights".
        inner = tf.matmul(x, weights)
        # check if relu should be applied
        if relu:
            return(tf.nn.relu(inner + biases))
        else:
            return inner + biases

def convolutional_layer(x, kernel, bias, reuse = True):
    conv = filter(x, kernel, conv_biasses[0],
                   relu=True, pool=True, norm=True, name="conv")

def fully_connected_layer(x, weights, bias):
    z = inner_product(x, weights, bias, name = 'fully_connected')

