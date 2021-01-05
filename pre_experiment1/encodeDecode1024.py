import tensorflow as tf
from tensorflow.keras import layers, Model


class EncodeDecode1024(Model):
    def __init__(self):
        super(EncodeDecode1024, self).__init__()
        # 1024*32 -> 1024*1024
        self.uppre = layers.UpSampling2D(size=(1, 32))

        # 1024 * 1024
        self.conv1 = layers.Conv2D(filters=8, kernel_size=3, strides=1, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.ac1 = layers.Activation('relu')
        # self.dp1 = layers.Dropout(rate=0.6)

        # 1024 -> 512
        self.mp1 = layers.MaxPool2D(pool_size=2, strides=2)

        self.conv2 = layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.ac2 = layers.Activation('relu')

        # 512 -> 256
        self.mp2 = layers.MaxPool2D(pool_size=2, strides=2)

        self.conv3 = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')
        self.bn3 = layers.BatchNormalization()
        self.ac3 = layers.Activation('relu')

        # 256 -> 128
        self.mp3 = layers.MaxPool2D(pool_size=2, strides=2)

        self.conv4 = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')
        self.bn4 = layers.BatchNormalization()
        self.ac4 = layers.Activation('relu')

        # 128 -> 64
        self.mp4 = layers.MaxPool2D(pool_size=2, strides=2)

        self.conv5 = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')
        self.bn5 = layers.BatchNormalization()
        self.ac5 = layers.Activation('relu')

        # 64 -> 32
        self.mp5 = layers.MaxPool2D(pool_size=2, strides=2)

        self.conv6 = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same')
        self.bn6 = layers.BatchNormalization()
        self.ac6 = layers.Activation('relu')

        # 32 -> 16
        self.mp6 = layers.MaxPool2D(pool_size=2, strides=2)

        self.conv7 = layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same')
        self.bn7 = layers.BatchNormalization()
        self.ac7 = layers.Activation('relu')

        self.conv8 = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')
        self.bn8 = layers.BatchNormalization()
        self.ac8 = layers.Activation('relu')

        # 16 -> 32
        self.up9 = layers.UpSampling2D(size=2)

        self.conv9 = layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same')
        self.bn9 = layers.BatchNormalization()
        self.ac9 = layers.Activation('relu')

        # 32 -> 64
        self.up10 = layers.UpSampling2D(size=2)

        self.conv10 = layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same')
        self.bn10 = layers.BatchNormalization()
        self.ac10 = layers.Activation('relu')

        # 64 -> 128
        self.up11 = layers.UpSampling2D(size=2)

        self.conv11 = layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same')
        self.bn11 = layers.BatchNormalization()
        self.ac11 = layers.Activation("relu")

        # 128 -> 256
        self.up12 = layers.UpSampling2D(size=2)

        self.conv12 = layers.Conv2D(filters=16, kernel_size=3, strides=1, padding='same')
        self.bn12 = layers.BatchNormalization()
        self.ac12 = layers.Activation("relu")

        # 256 -> 512
        self.up13 = layers.UpSampling2D(size=2)

        self.conv13 = layers.Conv2D(filters=8, kernel_size=3, strides=1, padding='same')
        self.bn13 = layers.BatchNormalization()
        self.ac13 = layers.Activation("relu")

        # 512 -> 1024
        self.up14 = layers.UpSampling2D(size=2)

        self.conv14 = layers.Conv2D(filters=2, kernel_size=3, strides=1, padding='same')
        self.bn14 = layers.BatchNormalization()
        self.ac14 = layers.Activation("softmax")

    def call(self, inputs, training=None, mask=None):
        # ->wh1024
        x = self.uppre(inputs)

        # 1024*1024*1 -> 1024*1024*8
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.ac1(x1)
        # 1024*124*8 => 512*512*8
        x1 = self.mp1(x1)

        # 512*512*8 => 512*512*16
        x2 = self.conv2(x1)
        x2 = self.bn2(x2)
        x2 = self.ac2(x2)
        # 512*512*16 => 256*256*16
        x2 = self.mp2(x2)

        # 256*256*16 = 256*256*32
        x3 = self.conv3(x2)
        x3 = self.bn3(x3)
        x3 = self.ac3(x3)
        # 256*256*32 => 128*128*32
        x3 = self.mp3(x3)

        # 128*128*32 = 128*128*64
        x4 = self.conv4(x3)
        x4 = self.bn4(x4)
        x4 = self.ac4(x4)
        # 128*128*64 => 64*64*64
        x4 = self.mp4(x4)

        # 64*64*64 = 64*64*128
        x5 = self.conv5(x4)
        x5 = self.bn5(x5)
        x5 = self.ac5(x5)
        # 64*64*128 => 32*32*128
        x5 = self.mp5(x5)

        #         # 32*32*128 = 32*32*256
        #         x6 = self.conv6(x5)
        #         x6 = self.bn6(x6)
        #         x6 = self.ac6(x6)
        #         # 32*32*256 => 16*16*256
        #         x6 = self.mp6(x6)

        #         # 16*16*256 = 16*16*512
        #         x7 = self.conv7(x6)
        #         x7 = self.bn7(x7)
        #         x7 = self.ac7(x7)

        # 16*16*256 = 16*16*512
        x7 = self.conv7(x5)
        x7 = self.bn7(x7)
        x7 = self.ac7(x7)

        # 16*16*512 => 16*16*256
        x8 = self.conv8(x7)
        x8 = self.bn8(x8)
        x9 = self.ac8(x8)

        #         # 16*16*512 => 16*16*256
        #         x8 = self.conv8(x7)
        #         x8 = self.bn8(x8)
        #         x8 = self.ac8(x8)

        #         # 16*16*256 => 32*32*256
        #         x9 = self.up9(x8)
        #         # 32*32*256 => 32*32*128
        #         x9 = self.conv9(x9)
        #         x9 = self.bn9(x9)
        #         x9 = layers.add([x9, x5])
        #         x9 = self.ac9(x9)

        # 32*32*128 => 64*64*128
        x10 = self.up10(x9)

        # 64*64*128 => 64*64*64
        x10 = self.conv10(x10)
        x10 = self.bn10(x10)
        x10 = layers.add([x10, x4])
        x10 = self.ac10(x10)

        # 64*64*64 => 128*128*64
        x11 = self.up11(x10)

        # 128*128*64 => 128*128*32
        x11 = self.conv11(x11)
        x11 = self.bn11(x11)
        x11 = layers.add([x11, x3])
        x11 = self.ac11(x11)

        # 128*128*32 => 256*256*32
        x12 = self.up12(x11)

        # 256*256*32 => 256*256*16
        x12 = self.conv12(x12)
        x12 = self.bn12(x12)
        x12 = layers.add([x12, x2])
        x12 = self.ac12(x12)

        # 256*256*16 => 512*512*16
        x13 = self.up13(x12)

        # 512*512*16 => 512*512*8
        x13 = self.conv13(x13)
        x13 = self.bn13(x13)
        x13 = layers.add([x13, x1])
        x13 = self.ac13(x13)

        # 512*512*8 => 1024*1024*8
        x14 = self.up14(x13)

        # 1024*1024*8 => 1024*1024*2
        x14 = self.conv14(x14)
        x14 = self.bn14(x14)
        # x14 = layers.add([x14, x1])
        x14 = self.ac14(x14)
        return x14,x,x1,x2,x3,x4,x5,x7,x8,x9,x10,x11,x12,x13
