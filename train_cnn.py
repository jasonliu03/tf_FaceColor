#coding=utf-8
import os
import sys
import tensorflow as tf

from records_utils import *

ITER_NUMS = 2000
EVAL_STEP = 50
LRARNING_RATE = 0.001
TRAIN_BATCH_SIZE = 10
KEEP_PROB = 0.9

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

            # calc test accuracy on large batch
            val, l = sess.run([test_img_batch, test_label_batch])
            l = tf.one_hot(l,CLASSES,1,0) 
            l = sess.run(l)
            print("test accuracy %g" % accuracy.eval(feed_dict={x: val, y_: l, keep_prob:1}))
    
    
            # save model paras
            saver = tf.train.Saver()  # defaults to saving all variables
            saver.save(sess, './model.ckpt')  #保存模型参数，注意把这里改为自己的路径

            coord.request_stop()
            coord.join(threads)


