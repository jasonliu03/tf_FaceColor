#coding=utf-8
import os
import sys
import tensorflow as tf
import numpy as np
from tensorflow.python.platform import gfile

from records_utils import *

ITER_NUMS = 30000
EVAL_STEP = 50
LRARNING_RATE = 0.01
DECAY_RATE = 0.1
TRAIN_BATCH_SIZE = 50
KEEP_PROB = 0.9
SAVE_MODEL_INTERVAL = 1000 

#Inception-v3模型瓶颈层的节点个数
BOTTLENECK_TENSOR_SIZE = 2048

#Inception-v3模型中代表瓶颈层结果的张量名称。在谷歌提供的Inception-v3模型中，这个张量
#名称就是'pool_3/reshape:0'。在训练的模型时，可以通过tensor.name来获取张量的名称。
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'

# note: image tensor name for input: --uint8 image [h,w,channels=3]
JPEG_DATA_TENSOR_NAME='DecodeJpeg:0'

MODEL_DIR = './'
MODEL_FILE = 'classify_image_graph_def.pb'


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

def get_random_cached_bottlenecks(
        sess,image_lists,label_lists,
        jpeg_data_tensor,bottleneck_tensor):
    bottlenecks=[]
    ground_truths=[]
    i=0;
    for img in image_lists:  
        label_name=label_lists[i]
        i=i+1
        bottleneck=get_or_create_bottleneck(
                    sess,img,
                    jpeg_data_tensor,bottleneck_tensor)
        ground_truth=np.zeros(2,dtype=np.float32)
        if label_name==1:
            ground_truth[1]=1.0
        else:
            ground_truth[0]=1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
    return bottlenecks,ground_truths



print os.path.join(MODEL_DIR,MODEL_FILE)
with gfile.FastGFile(os.path.join(MODEL_DIR,MODEL_FILE),'rb') as f:
    graph_def=tf.GraphDef()
    graph_def.ParseFromString(f.read())
#加载读取的Inception-v3模型，并返回数据输入所对应的张量以及计算瓶颈层
#结果所对应的张量
bottleneck_tensor,jpeg_data_tensor=tf.import_graph_def(
        graph_def,
        return_elements=[BOTTLENECK_TENSOR_NAME,JPEG_DATA_TENSOR_NAME])

tf.add_to_collection('bottleneck_tensor', bottleneck_tensor)
tf.add_to_collection('jpeg_data_tensor', jpeg_data_tensor)

# create the model
x=tf.placeholder(tf.float32,[None, BOTTLENECK_TENSOR_SIZE], name='x')
y_=tf.placeholder(tf.float32, [None,CLASSES], name='y_')

with tf.name_scope('final_training_ops'):
    weights=tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, CLASSES], stddev=0.001))
    biases=tf.Variable(tf.zeros([CLASSES]))
    y=tf.matmul(x,weights)+biases
    tf.add_to_collection('y', y)


cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

global_step = tf.Variable(0, trainable=False)
#lr = tf.train.exponential_decay(LRARNING_RATE, global_step, 1000, DECAY_RATE, staircase=True)
#train_step = tf.train.GradientDescentOptimizer(LRARNING_RATE).minimize(cross_entropy)
train_step = tf.train.AdamOptimizer(LRARNING_RATE).minimize(cross_entropy, global_step = global_step)
#train_step = tf.train.MomentumOptimizer(learning_rate=LRARNING_RATE,momentum= 0.9).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1] == '1':
        data = create_record("train.tfrecords")
    else:
        img, label = read_and_decode_forGenImage("train.tfrecords")
        #使用shuffle_batch可以随机打乱输入 next_batch挨着往下取
        # shuffle_batch才能实现[img,label]的同步,也即特征和label的同步,不然可能输入的特征和label不匹配
        # 比如只有这样使用,才能使img和label一一对应,每次提取一个image和对应的label
        # shuffle_batch返回的值就是RandomShuffleQueue.dequeue_many()的结果
        # Shuffle_batch构建了一个RandomShuffleQueue，并不断地把单个的[img,label],送入队列中
        img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=TRAIN_BATCH_SIZE, capacity=2000,
                                                    min_after_dequeue=1000)
        # large batch for test accuracy
        test_img_batch, test_label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=100, capacity=2000,
                                                    min_after_dequeue=1000)

        # 初始化所有的op
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            # 启动队列
            coord=tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            tf.summary.scalar('cross_entropy', cross_entropy)
            merged_summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter('logs/faceColor_logs',sess.graph)

            saver = tf.train.Saver(max_to_keep=0)  # defaults to saving all variables

            for i in range(ITER_NUMS):
              val, l = sess.run([img_batch, label_batch])

              val,l=get_random_cached_bottlenecks(
                        sess,val,l,jpeg_data_tensor,bottleneck_tensor)
                
              if i%EVAL_STEP == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x:val, y_: l})
                loss = cross_entropy.eval(feed_dict={x: val, y_: l})
                print("step %d, training accuracy %g"%(i, train_accuracy))
                print("loss %g"% loss)


              train_step.run(feed_dict={x: val, y_: l})

              # add summary
              summary_str = sess.run(merged_summary_op,feed_dict={x: val, y_: l})
              summary_writer.add_summary(summary_str, i)

              if (i+1) % SAVE_MODEL_INTERVAL == 0:
                  print("save model:%d" % (i+1))
                  saver.save(sess, './inceptionV3/model.ckpt', global_step = i+1)  #保存模型参数，注意把这里改为自己的路径
                  

            # calc test accuracy on large batch
            val, l = sess.run([test_img_batch, test_label_batch])
            val,l=get_random_cached_bottlenecks(
                        sess,val,l,jpeg_data_tensor,bottleneck_tensor)
            print("test accuracy %g" % accuracy.eval(feed_dict={x: val, y_: l}))
    
            coord.request_stop()
            coord.join(threads)


