#coding=utf-8
import os
import sys
import tensorflow as tf

from records_utils import *

ITER_NUMS = 8000
EVAL_STEP = 50
LRARNING_RATE = 0.001
TRAIN_BATCH_SIZE = 30
KEEP_PROB = 0.9
SAVE_MODEL_INTERVAL = 1000 

# weight initialization
def print_layer(t):
    print t.op.name, ' ', t.get_shape().as_list(), '\n'

def fc(x, n_out, name, fineturn=False, xavier=False):
    n_in = x.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        if fineturn:
            '''
            weight = tf.Variable(tf.constant(data_dict[name][0]), name="weights")
            bias = tf.Variable(tf.constant(data_dict[name][1]), name="bias")
            '''
            weight = tf.constant(data_dict[name][0], name="weights")
            bias = tf.constant(data_dict[name][1], name="bias")
            print "fineturn"
        elif not xavier:
            weight = tf.Variable(tf.truncated_normal([n_in, n_out], stddev=0.01), name='weights')
            bias = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[n_out]), 
                                                trainable=True, 
                                                name='bias')
            print "truncated_normal"
        else:
            weight = tf.get_variable(scope+'weights', shape=[n_in, n_out], 
                                                dtype=tf.float32,
                                                initializer=tf.contrib.layers.xavier_initializer_conv2d())
            bias = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[n_out]), 
                                                trainable=True, 
                                                name='bias')
            print "xavier"
        # 全连接层可以使用relu_layer函数比较方便，不用像卷积层使用relu函数
        activation = tf.nn.relu_layer(x, weight, bias, name=name)
        print_layer(activation)
        return activation


# create the model
x = tf.placeholder(tf.float32, shape=[None, WIDTH, HEIGHT, CHANNEL], name="x")
y_ = tf.placeholder(tf.float32, shape=[None, CLASSES])
keep_prob = tf.placeholder("float", name="keep_prob")

import vgg16
vgg = vgg16.Vgg16()
vgg.build(x)
feature_map = vgg.pool5

flatten  = tf.reshape(feature_map, [-1, 7*7*512])
fc6      = fc(flatten, 4096, 'fc6', xavier=True)
dropout1 = tf.nn.dropout(fc6, keep_prob)

fc7      = fc(dropout1, 4096, 'fc7', xavier=True)
dropout2 = tf.nn.dropout(fc7, keep_prob)
    
y = fc(dropout2, CLASSES, 'fc8', xavier=True)
tf.add_to_collection('pred_network', y)


cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(LRARNING_RATE).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1] == '1':
        data = create_record("train.tfrecords")
    else:
        img, label = read_and_decode("train.tfrecords")
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
              l = tf.one_hot(l,CLASSES,1,0) 
              l = sess.run(l)

              if i%EVAL_STEP == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x:val, y_: l, keep_prob:1})
                print("step %d, training accuracy %g"%(i, train_accuracy))
              train_step.run(feed_dict={x: val, y_: l, keep_prob:KEEP_PROB})

              # add summary
              summary_str = sess.run(merged_summary_op,feed_dict={x: val, y_: l, keep_prob:1})
              summary_writer.add_summary(summary_str, i)

              if (i+1) % SAVE_MODEL_INTERVAL == 0:
                  print("save model:%d" % (i+1))
                  saver.save(sess, './model.ckpt', global_step = i+1)  #保存模型参数，注意把这里改为自己的路径
                  

            # calc test accuracy on large batch
            val, l = sess.run([test_img_batch, test_label_batch])
            l = tf.one_hot(l,CLASSES,1,0) 
            l = sess.run(l)
            print("test accuracy %g" % accuracy.eval(feed_dict={x: val, y_: l, keep_prob:1}))
    
            coord.request_stop()
            coord.join(threads)


