#coding=utf-8
import os
import sys
import tensorflow as tf
import alexnet

from records_utils import *

ITER_NUMS = 10000
EVAL_STEP = 50
LRARNING_RATE = 0.001
DECAY_RATE = 0.1
TRAIN_BATCH_SIZE = 50
KEEP_PROB = 0.9
SAVE_MODEL_INTERVAL = 1000 


# create the model
x = tf.placeholder(tf.float32, shape=[None, WIDTH, HEIGHT, CHANNEL], name="x")
y_ = tf.placeholder(tf.float32, shape=[None, CLASSES], name='y_')
keep_prob = tf.placeholder("float", name="keep_prob")
    
y = alexnet.alex_net(x, keep_prob, CLASSES, WIDTH, CHANNEL)
tf.add_to_collection('pred_network', y)


cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

global_step = tf.Variable(0, trainable=False)
lr = tf.train.exponential_decay(LRARNING_RATE, global_step, 1000, DECAY_RATE, staircase=True)
#train_step = tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy, global_step = global_step)
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy, global_step = global_step)

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


