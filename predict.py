#coding=utf-8
from PIL import Image, ImageFilter
import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy as np

from glob import glob

WIDTH = 200
HEIGHT = 200
CHANNEL = 3 
CLASSES = 3 

def imageprepare(file_name):
    """ 
    This function returns the pixel values.
    The input is a png file location.
    """
    #in terminal 'mogrify -format png *.jpg' convert jpg to png
    im = Image.open(file_name)

#    plt.imshow(im)
#    plt.show()

    tv = np.asarray(im)
    tv = np.reshape(tv, [WIDTH*HEIGHT*CHANNEL])
    tv = np.true_divide(tv, 255) - 0.5
    return tv 


# weight initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

# convolution
def conv2d(x, W): 
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
# pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# create the model
x = tf.placeholder(tf.float32, shape=[None, WIDTH*HEIGHT*CHANNEL])
y_ = tf.placeholder(tf.float32, shape=[None, CLASSES])
W = tf.Variable(tf.zeros([WIDTH*HEIGHT*CHANNEL,CLASSES]))
b = tf.Variable(tf.zeros([CLASSES]))

# first convolutinal layer
w_conv1 = weight_variable([5, 5, CHANNEL, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, WIDTH, HEIGHT, CHANNEL])
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second convolutional layer
w_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# densely connected layer
w_fc1 = weight_variable([WIDTH/4*HEIGHT/4*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, WIDTH/4*HEIGHT/4*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
print h_fc1
# dropout
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
print h_fc1_drop
# readout layer
w_fc2 = weight_variable([1024, CLASSES])
b_fc2 = bias_variable([CLASSES])

#y = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)
y = tf.matmul(h_fc1_drop, w_fc2) + b_fc2



"""
Load the model2.ckpt file
file is stored in the same directory as this python script is started
Use the model to predict the integer. Integer is returend as list.

Based on the documentatoin at
https://www.tensorflow.org/versions/master/how_tos/variables/index.html
"""
saver = tf.train.Saver()
init_op = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init_op)
    saver.restore(sess, "./model.ckpt")#这里使用了之前保存的模型参数
    print ("Model restored.")

    total, right = 0, 0
    wrong = []
    for fn in sorted(glob('train_genpics/*_Label_*.png')):
        print('file_name:%s' % fn) 
        result=imageprepare(file_name=fn)

        prediction=tf.argmax(y,1)
        predint=prediction.eval(feed_dict={x: [result], keep_prob: 1}, session=sess)

        print('recognize result:%d' % predint[0])
        total += 1
        right += 1 if (str(predint[0]) == fn.split('.')[0][-1]) else 0
        if (str(predint[0]) != fn.split('.')[0][-1]):
            wrong.append(fn) 
    precision = 1.0 * right / total
    print('precision: %f' % precision)
    print('wrong cases:', wrong)
