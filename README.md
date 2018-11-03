# tf_FaceColor
classfy framework by tfrecord

1. config records_utils.py
WIDTH = 224
HEIGHT = 224
CHANNEL = 3
CLASSES = 2
classes = "man woman".split()

2. create tfrecord file
cd datasets/samples/xxx
python ../../../records_utils.py

3. train (only train_inceptionV3_rawimg need to resize image)
python train_cnn.py  (or python train.py with fc)
python train_alexnet.py
python train_vgg.py (default load vgg16.npy, adjust code vgg16->vgg19 to apply vgg19.npy)
python train_vgg_trainable.py （config FINETUNE, TRAINABLE in vgg19_trainable.py first)

python train_reload.py (default reload model.kept-1000, configure RELOAD_BASE first)

## inceptionV3 ##
python train_inceptionV3.py
python train_reload_inceptionV3.py

python train_inceptionV3_rawimg.py


3. predict (adjust size of the image to test first; rename to the right format xxx_Label_x)
python predict.py  (default: train_genpics)

python predict_inceptionV3.py
python predict_inceptionV3_rawimg.py
