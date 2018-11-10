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

#ÌØÕ÷ÏòÁ¿ save path£¨Ò»¸öÑµÁ·Êı¾İ»á±»¶à´ÎÊ¹ÓÃ£¬ÃâÈ¥ÖØ¸´¼ÆËãÌØÕ÷ÏòÁ¿£©
CACHE_DIR = './datasets/bottleneck'
#Êı¾İpath£¨Ã¿¸ö×ÓÎÄ¼ş¼ĞÖĞ´æ·ÅÍ¬Ò»Àà±ğµÄÍ¼Æ¬£©
INPUT_DATA = './datasets/test/Gender'

#ÑéÖ¤Êı¾İ percentage
VALIDATION_PERCENTAGE = 0
#²âÊÔÊı¾İ percentage
TEST_PERCENTAGE = 100


#°ÑÑù±¾ÖĞËùÓĞµÄÍ¼Æ¬ÁĞ±í²¢°´ÑµÁ·¡¢ÑéÖ¤¡¢²âÊÔÊı¾İ·Ö¿ª
def create_image_lists(testing_percentage, validation_percentage):
    # keyÎªÀà±ğ£¬valueÎª×Öµä£¨´æ´¢ÁËËùÓĞÍ¼Æ¬Ãû³Æ£©
    result = {}
    # »ñÈ¡µ±Ç°ÎÄ¼ş¼ĞÖ®ÏÂËùÓĞ×ÓÄ¿Â¼
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    # µÃµ½µÄµÚÒ»¸öÄ¿Â¼ÊÇµ±Ç°×ÓÄ¿Â¼£¬²»ĞèÒª¿¼ÂÇ
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        # »ñÈ¡µ±Ç°Ä¿Â¼ÏÂËùÓĞµÄÓĞĞ§Í¼Æ¬ÎÄ¼ş
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list: continue

        # Í¨¹ıÄ¿Â¼Ãû»ñÈ¡Àà±ğÃû
        label_name = dir_name.lower()
        
        # ³õÊ¼»¯
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            
            # Ëæ»ú»®·ÖÊı¾İ
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

#º¯ÊıÍ¨¹ıÀà±ğÃû³Æ¡¢ËùÊôÊı¾İ¼¯ºÍÍ¼Æ¬±àºÅ»ñÈ¡Ò»ÕÅÍ¼Æ¬µÄµØÖ·
# image_lists£ºËùÓĞÍ¼Æ¬ĞÅÏ¢
# image_dir£º¸ùÄ¿Â¼
# label_name£ºÀà±ğÃû³Æ
# index£ºÍ¼Æ¬±àºÅ
# category£º¸ÃÍ¼Æ¬ÊôÓÚÑµÁ·¼¯¡¢²âÊÔ¼¯orÑéÖ¤¼¯
def get_image_path(image_lists, image_dir, label_name, index, category):
    # »ñÈ¡¸ø¶¨Àà±ğÖĞËùÓĞÍ¼Æ¬µÄĞÅÏ¢
    label_lists = image_lists[label_name]
    # ¸ù¾İÊı¾İ¼¯Àà±ğ»ñÈ¡È«²¿Í¼Æ¬ĞÅÏ¢
    category_list = label_lists[category]
    mod_index = index % len(category_list)
    # »ñÈ¡Í¼Æ¬Â·¾¶
    base_name = category_list[mod_index]
    sub_dir = label_lists['dir']
    full_path = os.path.join(image_dir, sub_dir, base_name)
    return full_path

#º¯Êı»ñÈ¡Inception-v3Ä£ĞÍ´¦ÀíÖ®ºóµÄÌØÕ÷ÏòÁ¿µÄÎÄ¼şµØÖ·
def get_bottleneck_path(image_lists, label_name, index, category):
    return get_image_path(image_lists, CACHE_DIR, label_name, index, category) + '.txt'

#º¯ÊıÊ¹ÓÃ¼ÓÔØµÄÑµÁ·ºÃµÄInception-v3Ä£ĞÍ´¦ÀíÒ»ÕÅÍ¼Æ¬£¬µÃµ½Õâ¸öÍ¼Æ¬µÄÌØÕ÷ÏòÁ¿¡£
def run_bottleneck_on_image(sess, image_data, image_data_tensor, bottleneck_tensor):
    bottleneck_values = sess.run(bottleneck_tensor, {image_data_tensor: image_data})
    # ÉÏĞĞ´úÂë¾í»ıÍøÂç´¦Àí½á¹ûÊÇÒ»¸öËÄÎ¬Êı×é£¬ÏÂĞĞ´úÂë½«½á¹ûÑ¹Ëõ³ÉÒ»¸öÌØÕ÷
    bottleneck_values = np.squeeze(bottleneck_values)
    return bottleneck_values

#º¯Êı»áÏÈÊÔÍ¼Ñ°ÕÒÒÑ¾­¼ÆËãÇÒ±£´æÏÂÀ´µÄÌØÕ÷ÏòÁ¿£¬Èç¹ûÕÒ²»µ½ÔòÏÈ¼ÆËãÕâ¸öÌØÕ÷ÏòÁ¿£¬È»ºó±£´æµ½ÎÄ¼ş
def get_or_create_bottleneck(sess, image_lists, label_name, index, category, jpeg_data_tensor, bottleneck_tensor):
    # »ñÈ¡Ò»ÕÅÍ¼Æ¬¶ÔÓ¦µÄÌØÕ÷ÏòÁ¿ÎÄ¼şÂ·¾¶
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(CACHE_DIR, sub_dir)
    if not os.path.exists(sub_dir_path): 
        os.makedirs(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index, category)
    # Èç¹ûÌØÕ÷ÏòÁ¿æ??¼ş²»´æÔÚ£¬Í¨¹ıIncep-V3¼ÆËãÖ®ºó´æÈë½á¹û
    if not os.path.exists(bottleneck_path):
        image_path = get_image_path(image_lists, INPUT_DATA, label_name, index, category)
        image_data = gfile.FastGFile(image_path, 'rb').read()
        bottleneck_values = run_bottleneck_on_image(sess, image_data, jpeg_data_tensor, bottleneck_tensor)

        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    else:
        # Ö±½Ó´ÓÎÄ¼şè??È¡¶ÔÓ¦ÌØÕ÷ÏòÁ¿
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]

    return bottleneck_values

#»ñÈ¡È«²¿µÄ²âÊÔÊı¾İ£¬²¢¼ÆËãÕıç??ÂÊ
def get_test_bottlenecks(sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    label_name_list = list(image_lists.keys())
    # Ã¶¾ÙËùÓĞÀà±ğºÍÃ¿¸öÀà±ğÖĞµÄ²âÊÔÍ¼Æ¬
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
    # ¶ÁÈ¡ËùÓĞÍ¼Æ¬
    image_lists = create_image_lists(TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
    n_classes = len(image_lists.keys())
    
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        ## load the graph and restore the params
        saver = tf.train.import_meta_graph('./inceptionV3_rawimg/model.ckpt-1000.meta')
        #saver.restore(sess,tf.train.latest_checkpoint('./inceptionV3_rawimg'))
        saver.restore(sess, "./inceptionV3_rawimg/model.ckpt-46000")#ÕâÀïÊ¹ÓÃÁËÖ®Ç°±£´æµÄÄ£ĞÍ²ÎÊı
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

        # ¼ÆËãÕıÈ·ÂÊ¡£
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            
        # ÔÚ×îºóµÄ²âÊÔÊı¾İÉÏ²âÊÔÕıÈ·ÂÊ¡£
        test_bottlenecks, test_ground_truth = get_test_bottlenecks(
            sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor)
        test_accuracy = sess.run(evaluation_step, feed_dict={
            x: test_bottlenecks, y_: test_ground_truth})
        print('Final test accuracy = %.1f%%' % (test_accuracy * 100))


if __name__ == '__main__':
    main()

