#coding=utf-8
import os
import tensorflow as tf

import sys

from records_utils import *


x = tf.placeholder(tf.float32, shape=[None, WIDTH*HEIGHT*CHANNEL])
y_ = tf.placeholder(tf.float32, shape=[None, CLASSES])
W = tf.Variable(tf.zeros([WIDTH*HEIGHT*CHANNEL,CLASSES]))
b = tf.Variable(tf.zeros([CLASSES]))

y = tf.matmul(x,W) + b

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1] == '1':
        data = create_record("train.tfrecords")
    else:
        img, label = read_and_decode("train.tfrecords")
        print("jason",img,label)
        #使用shuffle_batch可以随机打乱输入 next_batch挨着往下取
        # shuffle_batch才能实现[img,label]的同步,也即特征和label的同步,不然可能输入的特征和label不匹配
        # 比如只有这样使用,才能使img和label一一对应,每次提取一个image和对应的label
        # shuffle_batch返回的值就是RandomShuffleQueue.dequeue_many()的结果
        # Shuffle_batch构建了一个RandomShuffleQueue，并不断地把单个的[img,label],送入队列中
        img_batch, label_batch = tf.train.shuffle_batch([img, label],
                                                    batch_size=10, capacity=2000,
                                                    min_after_dequeue=1000)

        # 初始化所有的op
        #init = tf.initialize_all_variables()
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            # 启动队列
            coord=tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            merged_summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter('logs/faceColor_1fc_logs',sess.graph)


            for i in range(2000):
              val, l = sess.run([img_batch, label_batch])
              l = tf.one_hot(l,CLASSES,1,0) 
              l = sess.run(l)

              if i%100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x:val, y_: l})
                print("step %d, training accuracy %g"%(i, train_accuracy))
              train_step.run(feed_dict={x: val, y_: l})
              summary_str = sess.run(merged_summary_op,feed_dict={x: val, y_: l})
              summary_writer.add_summary(summary_str, i)

            val, l = sess.run([img_batch, label_batch])
            l = tf.one_hot(l,CLASSES,1,0) 
            l = sess.run(l)
            print("test accuracy %g" % accuracy.eval(feed_dict={x: val, y_: l}))
    
    
            saver = tf.train.Saver()  # defaults to saving all variables
            saver.save(sess, './model.ckpt')  #保存模型参数，注意把这里改为自己的路径

            coord.request_stop()
            coord.join(threads)


