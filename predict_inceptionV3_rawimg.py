# -*- coding: utf-8 -*-
"""
Created on Nov 1 2018
@author: jason.liu
"""

import glob
import os.path
import random
import numpy as np
import tensorflow as tf

#特征向量 save path（一个训练数据会被多次使用，免去重复计算特征向量）
CACHE_DIR = './datasets/bottleneck'
#数据path（每个子文件夹中存放同一类别的图片）
INPUT_DATA = './datasets/test/Gender'

#验证数据 percentage
VALIDATION_PERCENTAGE = 0
#测试数据 percentage
TEST_PERCENTAGE = 100


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
    
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        ## load the graph and restore the params
        saver = tf.train.import_meta_graph('./inceptionV3_rawimg/model.ckpt-1000.meta')
        #saver.restore(sess,tf.train.latest_checkpoint('./inceptionV3_rawimg'))
        saver.restore(sess, "./inceptionV3_rawimg/model.ckpt-46000")#这里使用了之前保存的模型参数
        print ("Model restored.")

        ## get the tensor and operation
        graph = tf.get_default_graph()
        #bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(
        #    graph, return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])
        #bottleneck_tensor = graph.get_tensor_by_name(BOTTLENECK_TENSOR_NAME)
        #jpeg_data_tensor = graph.get_tensor_by_name(JPEG_DATA_TENSOR_NAME)
        bottleneck_tensor = tf.get_collection("bottleneck_tensor")[0]
        jpeg_data_tensor = tf.get_collection("jpeg_data_tensor")[0]
        x=graph.get_operation_by_name('x').outputs[0]
        y_ = graph.get_tensor_by_name('y_:0')
        y=tf.get_collection("y")[0]

        # 计算正确率。
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            
        # 在最后的测试数据上测试正确率。
        test_bottlenecks, test_ground_truth = get_test_bottlenecks(
            sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor)
        test_accuracy = sess.run(evaluation_step, feed_dict={
            x: test_bottlenecks, y_: test_ground_truth})
        print('Final test accuracy = %.1f%%' % (test_accuracy * 100))


if __name__ == '__main__':
    main()

