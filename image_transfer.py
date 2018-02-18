#coding=utf-8
import os
import sys
import tensorflow as tf
from PIL import Image

from records_utils import *

# to change:
SAMPLE_NUMS = 49+48+39+87

def image_gen(tfrecordsFile, savePath):
    batch = read_and_decode_forGenImage(tfrecordsFile)

    # 初始化所有的op
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        # 启动队列
        coord=tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(SAMPLE_NUMS):
            example,l = sess.run(batch)#take out image and label
            img=Image.fromarray(example, 'RGB')
            img.save(savePath+str(i)+'_''Label_'+str(l)+'.png')#save image

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1] == 'test':
        image_gen("test.tfrecords", "./test_genpics/")
    else:
        image_gen("train.tfrecords", "./train_genpics/")
