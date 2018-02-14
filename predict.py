#coding=utf-8
from PIL import Image, ImageFilter
import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy as np

WIDTH = 64
HEIGHT = 64
CHANNEL = 3 
CLASSES = 4 

def imageprepare():
    """ 
    This function returns the pixel values.
    The imput is a png file location.
    """
    file_name='./genpics/108_Label_1.png'#导入自己的图片地址
    #in terminal 'mogrify -format png *.jpg' convert jpg to png
    im = Image.open(file_name)

    plt.imshow(im)
    plt.show()

    data = im.getdata()
    print("im.shape:", np.asarray(im).shape)
    tv = list(im.getdata()) #get pixel values

    tv = np.asarray(im)
    tv = np.reshape(tv, [WIDTH*HEIGHT*CHANNEL])
    print("tv:", tv)
    print("----------------------------------------")
    print("image.tobytes", im.tobytes())
    return tv 



    """ 
    This function returns the predicted integer.
    The imput is the pixel values from the imageprepare() function.
    """

    # Define the model (same as when creating the model file)
result=imageprepare()
print("result.shape:", result.shape)
x = tf.placeholder(tf.float32, shape=[None, WIDTH*HEIGHT*CHANNEL])
W = tf.Variable(tf.zeros([WIDTH*HEIGHT*CHANNEL,CLASSES]))
b = tf.Variable(tf.zeros([CLASSES]))


y_pred=tf.nn.softmax(tf.matmul(x, W) + b)

init_op = tf.initialize_all_variables()



"""
Load the model2.ckpt file
file is stored in the same directory as this python script is started
Use the model to predict the integer. Integer is returend as list.

Based on the documentatoin at
https://www.tensorflow.org/versions/master/how_tos/variables/index.html
"""
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init_op)
    saver.restore(sess, "./model.ckpt")#这里使用了之前保存的模型参数
    print ("Model restored.")
    print ("W:", W)
    print ("b:", b)

    prediction=tf.argmax(y_pred,1)
    predint=prediction.eval(feed_dict={x: [result]}, session=sess)

    print('recognize result:')
    print(predint[0])
