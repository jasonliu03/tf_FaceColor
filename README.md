# tf_FaceColor
classfy framework by tfrecord

1. config records_utils.py
WIDTH = 200
HEIGHT = 200
CHANNEL = 3
CLASSES = 3
classes = "1 2 3".split()

2. create tfrecord file
python records_utils.py

2. train (not need to normalized image)
python train_cnn.py  (or python train.py with fc)

3. predict (adjust size of the image to test first; rename to the right format xxx_Label_x)
python predict.py  (default: train_genpics)
