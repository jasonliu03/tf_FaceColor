#coding=utf-8
import os
import tensorflow as tf
from PIL import Image

cwd = os.getcwd()

WIDTH = 64
HEIGHT = 64
CHANNEL = 3 
CLASSES = 4 

classes = ['red-49','yellow-48','cyan-39','normal-87']
#制作二进制数据
def create_record(filename):
    writer = tf.python_io.TFRecordWriter(filename)
    for index, name in enumerate(classes):
        class_path = cwd +"/"+ name+"/"
        for img_name in sorted(os.listdir(class_path)):
            img_path = class_path + img_name
            img = Image.open(img_path)
            img = img.resize((WIDTH, HEIGHT))
            img_raw = img.tobytes() #将图片转化为原生bytes
            #print(index,img_raw)
            example = tf.train.Example(
               features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
               }))
            writer.write(example.SerializeToString())
    writer.close()

#读取二进制数据
def read_and_decode(filename):
    # 创建文件队列,不限读取的数量
    filename_queue = tf.train.string_input_producer([filename])
    # create a reader from file queue
    reader = tf.TFRecordReader()
    # reader从文件队列中读入一个序列化的样本
    _, serialized_example = reader.read(filename_queue) # get feature from serialized example
    # 解析符号化的样本
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'img_raw': tf.FixedLenFeature([], tf.string)
        }   
    )   
    label = features['label']
    img = features['img_raw']
    img = tf.decode_raw(img, tf.uint8)
    img = tf.reshape(img, [WIDTH*HEIGHT*CHANNEL])
    #img = tf.reshape(img, [64, 64, 3]) 
    #img = tf.cast(img, tf.float32) * (1. / 255) - 0.5 
    label = tf.cast(label, tf.int32)
    return img, label

