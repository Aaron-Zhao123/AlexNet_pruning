import cv2
import tensorflow as tf
import argparse
import numpy as np
import multiprocessing
import msgpack
import os
import sys

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.tfutils.varreplace import remap_variables

import alexnet_model


def get_data(dataset_name, BATCH_SIZE):
    isTrain = dataset_name == 'train'
    ds = dataset.ILSVRC12(args.data, dataset_name, shuffle=isTrain)

    meta = dataset.ILSVRCMeta()
    pp_mean = meta.get_per_pixel_mean()
    pp_mean_224 = pp_mean[16:-16, 16:-16, :]

    if isTrain:
        class Resize(imgaug.ImageAugmentor):
            def __init__(self):
                self._init(locals())

            def _augment(self, img, _):
                h, w = img.shape[:2]
                size = 224
                scale = self.rng.randint(size, 308) * 1.0 / min(h, w)
                scaleX = scale * self.rng.uniform(0.85, 1.15)
                scaleY = scale * self.rng.uniform(0.85, 1.15)
                desSize = map(int, (max(size, min(w, scaleX * w)),
                                    max(size, min(h, scaleY * h))))
                dst = cv2.resize(img, tuple(desSize),
                                 interpolation=cv2.INTER_CUBIC)
                return dst

        augmentors = [
            Resize(),
            imgaug.Rotation(max_deg=10),
            imgaug.RandomApplyAug(imgaug.GaussianBlur(3), 0.5),
            imgaug.Brightness(30, True),
            imgaug.Gamma(),
            imgaug.Contrast((0.8, 1.2), True),
            imgaug.RandomCrop((224, 224)),
            imgaug.RandomApplyAug(imgaug.JpegNoise(), 0.8),
            imgaug.RandomApplyAug(imgaug.GaussianDeform(
                [(0.2, 0.2), (0.2, 0.8), (0.8, 0.8), (0.8, 0.2)],
                (224, 224), 0.2, 3), 0.1),
            imgaug.Flip(horiz=True),
            imgaug.MapImage(lambda x: x - pp_mean_224),
        ]
    else:
        def resize_func(im):
            h, w = im.shape[:2]
            scale = 256.0 / min(h, w)
            desSize = map(int, (max(224, min(w, scale * w)),
                                max(224, min(h, scale * h))))
            im = cv2.resize(im, tuple(desSize), interpolation=cv2.INTER_CUBIC)
            return im
        augmentors = [
            imgaug.MapImage(resize_func),
            imgaug.CenterCrop((224, 224)),
            imgaug.MapImage(lambda x: x - pp_mean_224),
        ]
    ds = AugmentImageComponent(ds, augmentors, copy=False)
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    if isTrain:
        ds = PrefetchDataZMQ(ds, min(12, multiprocessing.cpu_count()))
    return ds

def test():
    global BATCH_SIZE
    BATCH_SIZE = 128
    isLoad = False

    data_train = get_data('train', BATCH_SIZE)
    # data_test = get_data('val', BATCH_SIZE)
    # data_test.reset_state()
    # generator = data_test.get_data()

    model = alexnet_model.alexnet(isLoad)
    inference(model)

def inference(model):
    """
    continue building the graph
    """
    logits = model.pred
    prob = tf.nn.softmax(logits)
    top5_error = model.error_rates(topk = 5)

    data_test = get_data('val', BATCH_SIZE)
    data_test.reset_state()
    generator = data_test.get_data()

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for dp in generator:
            # print(np.shape(dp[0]))
            # print(np.shape(dp[1]))
            top5_val = sess.run(top5_error, feed_dict = {
                model.images:dp[0],
                model.labels:dp[1],
                model.isTrain: False
            })



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='ids of GPUS')
    parser.add_argument('--data', help='imagenet dir')
    parser.add_argument('--load', help='load a model, .npy')
    args = parser.parse_args()
    if (args.gpu):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # NR_GPU = len(args.gpu.split(','))
    # TOTAL_BATCH_SIZE = 128
    # global BATCH_SIZE
    # BATCH_SIZE = TOTAL_BATCH_SIZE // NR_GPU

    test()
