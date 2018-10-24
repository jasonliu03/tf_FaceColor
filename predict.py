#coding=utf-8
from PIL import Image, ImageFilter
import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy as np

from glob import glob

from records_utils import WIDTH, HEIGHT, CHANNEL, CLASSES

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
    tv = np.reshape(tv, [WIDTH, HEIGHT, CHANNEL])
    tv = np.true_divide(tv, 255) - 0.5
    return tv 

init_op = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init_op)

    ## load the graph and restore the params
    saver = tf.train.import_meta_graph('model.ckpt-1000.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./'))
    #saver.restore(sess, "./model.ckpt-1000")#这里使用了之前保存的模型参数
    print ("Model restored.")

    ## get the tensor and operation
    graph = tf.get_default_graph()
    x=graph.get_operation_by_name('x').outputs[0]
    #x=graph.get_tensor_by_name('x:0')
    keep_prob=graph.get_tensor_by_name('keep_prob:0')
    y=tf.get_collection("pred_network")[0]

    total, right = 0, 0
    wrong = []
    for fn in sorted(glob('train_genpics/testGentle/test03/*_Label_*.jpg')):
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
