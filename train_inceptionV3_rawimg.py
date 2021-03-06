# -*- coding: utf-8 -*-

import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import time

#模型和样本路径的设置
#inception-V3瓶颈层节点个数
BOTTLENECK_TENSOR_SIZE = 2048
#瓶颈层tenbsor name
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
#图像输入tensor name
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'

# v3 模型的路径
MODEL_DIR = './'
# v3 模型文件名
MODEL_FILE= 'classify_image_graph_def.pb'

#特征向量 save path（一个训练数据会被多次使用，免去重复计算特征向量）
CACHE_DIR = './datasets/bottleneck'
#数据path（每个子文件夹中存放同一类别的图片）
INPUT_DATA = './datasets/samples/Gender'

#验证数据 percentage
VALIDATION_PERCENTAGE = 10
#测试数据 percentage
TEST_PERCENTAGE = 10

#神经网络参数的设置
LEARNING_RATE = 0.01
STEPS = 60000
BATCH = 50
SAVE_MODEL_INTERVAL = 1000

#把样本中所有的图片列表并按训练、验证、测试数据分开
def create_image_lists(testing_percentage, validation_percentage):
    # key为类别，value为字典（存储了所有图片名称）
    result = {}
    # 获取当前文件夹之下所有子目录
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    # 得到的第一个目录是当前子目录，不需要考虑
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        # 获取当前目录下所有的有效图片文件
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list: continue

        # 通过目录名获取类别名
        label_name = dir_name.lower()
        
        # 初始化
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            
            # 随机划分数据
            chance = np.random.randint(100)
            if chance < validation_percentage:
                validation_images.append(base_name)
            elif chance < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)

        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
            }
    return result

#函数通过类别名称、所属数据集和图片编号获取一张图片的地址
# image_lists：所有图片信息
# image_dir：根目录
# label_name：类别名称
# index：图片编号
# category：该图片属于训练集、测试集or验证集
def get_image_path(image_lists, image_dir, label_name, index, category):
    # 获取给定类别中所有图片的信息
    label_lists = image_lists[label_name]
    # 根据数据集类别获取全部图片信息
    category_list = label_lists[category]
    mod_index = index % len(category_list)
    # 获取图片路径
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path

#函数获取Inception-v3模型处理之后的特征向量的文件地址
def get_bottleneck_path(image_lists, label_name, index, category):
    return get_image_path(image_lists, CACHE_DIR, label_name, index, category) + '.txt'

#函数使用加载的训练好的Inception-v3模型处理一张图片，得到这个图片的特征向量。
def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
    # 上行代码卷积网络处理结果是一个四维数组，下行代码将结果压缩成一个特征
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values

#函数会先试图寻找已经计算且保存下来的特征向量，如果找不到则先计算这个特征向量，然后保存到文件
def get_or_create_bottleneck(sess, image_lists, label_name, index, category, jpeg_data_tensor, bottleneck_tensor):
    # 获取一张图片对应的特征向量文件路径
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(CACHE_DIR, sub_dir)
    if not os.path.exists(sub_dir_path): 
        os.makedirs(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index, category)
    # 如果特征向量�??件不存在，通过Incep-V3计算之后存入结果
    if not os.path.exists(bottleneck_path):
        image_path = get_image_path(image_lists, INPUT_DATA, label_name, index, category)
        image_data = gfile.FastGFile(image_path, 'rb').read()
        bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    else:
        # 直接从文件�??取对应特征向量
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]

    return bottleneck_values


# 函数随机�??取一个batch的图片作为训练数据
# how_many：一个batch图片的数閲??
def get_random_cached_bottlenecks(sess, n_classes, image_lists, how_many, category, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    for _ in range(how_many):
        label_index = random.randrange(n_classes)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(65536)
        bottleneck = get_or_create_bottleneck(
            sess, image_lists, label_name, image_index, category, jpeg_data_tensor, bottleneck_tensor)
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)

    return bottlenecks, ground_truths

#获取全部的测试数据，并计算正�??率
def get_test_bottlenecks(sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    label_name_list = list(image_lists.keys())
    # 枚举所有类别和每个类别中的测试图片
    for label_index, label_name in enumerate(label_name_list):
        category = 'testing'
        for index, unused_base_name in enumerate(image_lists[label_name][category]):
            bottleneck = get_or_create_bottleneck(sess, image_lists, label_name, index, category,jpeg_data_tensor, bottleneck_tensor)
            ground_truth = np.zeros(n_classes, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
    return bottlenecks, ground_truths


def main():
    # 读取所有图片
    image_lists = create_image_lists(TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
    n_classes = len(image_lists.keys())
    
    # 读取宸????训练好的Inception-v3模型。
    with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    start = time.time()
    # 加载Incep-V3模型，并返回数据输入所对搴????tensor及计算瓶颈层结果的tensor
    bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(
        graph_def, return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])
    print("耗时："+str(time.time()-start))
    tf.add_to_collection('bottleneck_tensor', bottleneck_tensor)
    tf.add_to_collection('jpeg_data_tensor', jpeg_data_tensor)

    # 定义新的神经网络输鍏????即图片经过Incep-V3之后的节点取值
    x = tf.placeholder(tf.float32, [None, BOTTLENECK_TENSOR_SIZE], name='x')
    
    # 定义新的标准答案的输入
    y_ = tf.placeholder(tf.float32, [None, n_classes], name='y_')
    
    # 定义一层新的全链接灞??    
    with tf.name_scope('final_training_ops'):
        weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, n_classes], stddev=0.001))
        biases = tf.Variable(tf.zeros([n_classes]))
        y = tf.matmul(x, weights) + biases
        tf.add_to_collection('y', y)
        final_tensor = tf.nn.softmax(y)
        
    # 定义交叉熵损失函数。
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    #train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)
    #train_step = tf.train.MomentumOptimizer(LEARNING_RATE, momentum=0.9).minimize(cross_entropy_mean)
    
    # 计算正确率。
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(y_, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        # 训练过程。
        saver = tf.train.Saver(max_to_keep=0)  # defaults to saving all variables
        for i in range(STEPS):
            # 每次获取一个batch的训练数据
            train_bottlenecks, train_ground_truth = get_random_cached_bottlenecks(
                sess, n_classes, image_lists, BATCH, 'training', jpeg_data_tensor, bottleneck_tensor)
            sess.run(train_step, feed_dict={x: train_bottlenecks, y_: train_ground_truth})

            if i % 100 == 0 or i + 1 == STEPS:
                validation_bottlenecks, validation_ground_truth = get_random_cached_bottlenecks(
                    sess, n_classes, image_lists, BATCH, 'validation', jpeg_data_tensor, bottleneck_tensor)
                validation_accuracy = sess.run(evaluation_step, feed_dict={
                    x: validation_bottlenecks, y_: validation_ground_truth})
                loss = sess.run(cross_entropy_mean, feed_dict={
                    x: validation_bottlenecks, y_: validation_ground_truth})
                print('Step %d: Validation accuracy on random sampled %d examples = %.1f%%' % (i, BATCH, validation_accuracy * 100))
                print('loss = %f' % loss)

            if (i+1) % SAVE_MODEL_INTERVAL == 0:
                print("save model:%d" % (i+1))
                saver.save(sess, './inceptionV3_rawimg/model.ckpt', global_step = i+1)  #保存模型参数，注意把这里改为自己的路径
            
        # 在最后的测试数据上测试正确率。
        test_bottlenecks, test_ground_truth = get_test_bottlenecks(
            sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor)
        test_accuracy = sess.run(evaluation_step, feed_dict={
            x: test_bottlenecks, y_: test_ground_truth})
        print('Final test accuracy = %.1f%%' % (test_accuracy * 100))


if __name__ == '__main__':
    main()

