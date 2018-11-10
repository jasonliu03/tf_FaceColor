# -*- coding: utf-8 -*-

import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import time

#ģ�ͺ�����·��������
#inception-V3ƿ����ڵ����
BOTTLENECK_TENSOR_SIZE = 2048
#ƿ����tenbsor name
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
#ͼ������tensor name
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'

# v3 ģ�͵�·��
MODEL_DIR = './'
# v3 ģ���ļ���
MODEL_FILE= 'classify_image_graph_def.pb'

#�������� save path��һ��ѵ�����ݻᱻ���ʹ�ã���ȥ�ظ���������������
CACHE_DIR = './datasets/bottleneck'
#����path��ÿ�����ļ����д��ͬһ����ͼƬ��
INPUT_DATA = './datasets/samples/Gender'

#��֤���� percentage
VALIDATION_PERCENTAGE = 10
#�������� percentage
TEST_PERCENTAGE = 10

#���������������
LEARNING_RATE = 0.01
STEPS = 60000
BATCH = 50
SAVE_MODEL_INTERVAL = 1000

#�����������е�ͼƬ�б���ѵ������֤���������ݷֿ�
def create_image_lists(testing_percentage, validation_percentage):
    # keyΪ���valueΪ�ֵ䣨�洢������ͼƬ���ƣ�
    result = {}
    # ��ȡ��ǰ�ļ���֮��������Ŀ¼
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    # �õ��ĵ�һ��Ŀ¼�ǵ�ǰ��Ŀ¼������Ҫ����
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        # ��ȡ��ǰĿ¼�����е���ЧͼƬ�ļ�
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list: continue

        # ͨ��Ŀ¼����ȡ�����
        label_name = dir_name.lower()
        
        # ��ʼ��
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            
            # �����������
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

#����ͨ��������ơ��������ݼ���ͼƬ��Ż�ȡһ��ͼƬ�ĵ�ַ
# image_lists������ͼƬ��Ϣ
# image_dir����Ŀ¼
# label_name���������
# index��ͼƬ���
# category����ͼƬ����ѵ���������Լ�or��֤��
def get_image_path(image_lists, image_dir, label_name, index, category):
    # ��ȡ�������������ͼƬ����Ϣ
    label_lists = image_lists[label_name]
    # �������ݼ�����ȡȫ��ͼƬ��Ϣ
    category_list = label_lists[category]
    mod_index = index % len(category_list)
    # ��ȡͼƬ·��
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path

#������ȡInception-v3ģ�ʹ���֮��������������ļ���ַ
def get_bottleneck_path(image_lists, label_name, index, category):
    return get_image_path(image_lists, CACHE_DIR, label_name, index, category) + '.txt'

#����ʹ�ü��ص�ѵ���õ�Inception-v3ģ�ʹ���һ��ͼƬ���õ����ͼƬ������������
def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
    # ���д��������紦������һ����ά���飬���д��뽫���ѹ����һ������
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values

#����������ͼѰ���Ѿ������ұ�����������������������Ҳ������ȼ����������������Ȼ�󱣴浽�ļ�
def get_or_create_bottleneck(sess, image_lists, label_name, index, category, jpeg_data_tensor, bottleneck_tensor):
    # ��ȡһ��ͼƬ��Ӧ�����������ļ�·��
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(CACHE_DIR, sub_dir)
    if not os.path.exists(sub_dir_path): 
        os.makedirs(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index, category)
    # ������������??�������ڣ�ͨ��Incep-V3����֮�������
    if not os.path.exists(bottleneck_path):
        image_path = get_image_path(image_lists, INPUT_DATA, label_name, index, category)
        image_data = gfile.FastGFile(image_path, 'rb').read()
        bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    else:
        # ֱ�Ӵ��ļ��??ȡ��Ӧ��������
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]

    return bottleneck_values


# ��������??ȡһ��batch��ͼƬ��Ϊѵ������
# how_many��һ��batchͼƬ�����??
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

#��ȡȫ���Ĳ������ݣ����������??��
def get_test_bottlenecks(sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    label_name_list = list(image_lists.keys())
    # ö����������ÿ������еĲ���ͼƬ
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
    # ��ȡ����ͼƬ
    image_lists = create_image_lists(TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
    n_classes = len(image_lists.keys())
    
    # ��ȡ�????ѵ���õ�Inception-v3ģ�͡�
    with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    start = time.time()
    # ����Incep-V3ģ�ͣ��������������������????tensor������ƿ��������tensor
    bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(
        graph_def, return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])
    print("��ʱ��"+str(time.time()-start))
    tf.add_to_collection('bottleneck_tensor', bottleneck_tensor)
    tf.add_to_collection('jpeg_data_tensor', jpeg_data_tensor)

    # �����µ����������????��ͼƬ����Incep-V3֮��Ľڵ�ȡֵ
    x = tf.placeholder(tf.float32, [None, BOTTLENECK_TENSOR_SIZE], name='x')
    
    # �����µı�׼�𰸵�����
    y_ = tf.placeholder(tf.float32, [None, n_classes], name='y_')
    
    # ����һ���µ�ȫ�����??    
    with tf.name_scope('final_training_ops'):
        weights = tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE, n_classes], stddev=0.001))
        biases = tf.Variable(tf.zeros([n_classes]))
        y = tf.matmul(x, weights) + biases
        tf.add_to_collection('y', y)
        final_tensor = tf.nn.softmax(y)
        
    # ���彻������ʧ������
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    #train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)
    train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy_mean)
    #train_step = tf.train.MomentumOptimizer(LEARNING_RATE, momentum=0.9).minimize(cross_entropy_mean)
    
    # ������ȷ�ʡ�
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(final_tensor, 1), tf.argmax(y_, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        # ѵ�����̡�
        saver = tf.train.Saver(max_to_keep=0)  # defaults to saving all variables
        for i in range(STEPS):
            # ÿ�λ�ȡһ��batch��ѵ������
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
                saver.save(sess, './inceptionV3_rawimg/model.ckpt', global_step = i+1)  #����ģ�Ͳ�����ע��������Ϊ�Լ���·��
            
        # �����Ĳ��������ϲ�����ȷ�ʡ�
        test_bottlenecks, test_ground_truth = get_test_bottlenecks(
            sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor)
        test_accuracy = sess.run(evaluation_step, feed_dict={
            x: test_bottlenecks, y_: test_ground_truth})
        print('Final test accuracy = %.1f%%' % (test_accuracy * 100))


if __name__ == '__main__':
    main()

