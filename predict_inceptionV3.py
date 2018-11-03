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
#    tv = np.true_divide(tv, 255) - 0.5
    return tv 



#这个函数使用加载的训练好的Inception-v3模型处理一张图片，得到这个图片的特征向量。
def run_bottleneck_on_image(sess,image_data,image_data_tensor,bottleneck_tensor):
    bottleneck_values=sess.run(bottleneck_tensor,{image_data_tensor:image_data})
    bottleneck_values=np.squeeze(bottleneck_values)
    return bottleneck_values

#这个函数获取一张图片经过Inception-v3模型处理之后的特征向量。这个函数会先试图寻找
#已经计算且保存下来的特征向量，如果找不到则先计算这个特征向量，然后保存到文件
def get_or_create_bottleneck(
        sess,image_data,jpeg_data_tensor,bottleneck_tensor):
    bottleneck_values=run_bottleneck_on_image(sess,image_data,jpeg_data_tensor,bottleneck_tensor)
    return bottleneck_values


init_op = tf.initialize_all_variables()
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
with tf.Session(config=config) as sess:
    sess.run(init_op)

    ## load the graph and restore the params
    saver = tf.train.import_meta_graph('./inceptionV3/model.ckpt-1000.meta')
    #saver.restore(sess,tf.train.latest_checkpoint('./'))
    saver.restore(sess, "./inceptionV3/model.ckpt-1000")#这里使用了之前保存的模型参数
    print ("Model restored.")

    ## get the tensor and operation
    graph = tf.get_default_graph()
    bottleneck_tensor = tf.get_collection("bottleneck_tensor")[0]
    jpeg_data_tensor = tf.get_collection("jpeg_data_tensor")[0]
    x=graph.get_operation_by_name('x').outputs[0]
    #x=graph.get_tensor_by_name('x:0')
    y=tf.get_collection("y")[0]

    total, right = 0, 0
    wrong = []
    for fn in sorted(glob('train_genpics/testGentle/test03/*_Label_*.jpg')):
        print('file_name:%s' % fn) 
        result=imageprepare(file_name=fn)

        val=get_or_create_bottleneck(sess,result,jpeg_data_tensor,bottleneck_tensor)

        prediction=tf.argmax(y,1)
        predint=prediction.eval(feed_dict={x: [val]}, session=sess)

        print('recognize result:%d' % predint[0])
        total += 1
        right += 1 if (str(predint[0]) == fn.split('.')[0][-1]) else 0
        if (str(predint[0]) != fn.split('.')[0][-1]):
            wrong.append(fn) 
    precision = 1.0 * right / total
    print('precision: %f' % precision)
    print('wrong cases:', wrong)
