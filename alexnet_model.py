import tensorflow as tf
import numpy as np
import pickle
import sys

"""
gives back self.pred, self.
"""
class alexnet(object):
    def __init__(self, isLoad):
        # image, label = inputs
        # image = image / 255.0
        # return two placeholders: self.images and self.labels
        self.get_placeholders()
        # intialize variables in their namescopes
        self._get_variables(isLoad)
        self._init_weight_masks(isLoad)
        self.conv_network()

    def error_rates(self,topk = 1):
        return tf.cast(tf.logical_not(tf.nn.in_top_k(self.pred, self.labels, topk)),
            tf.float32)

    def conv_network(self):
        imgs = self.images

        conv1 = self.conv_layer(imgs, 'conv1', padding = 'VALID', stride = 4, prune = True)
        pool1 = self.maxpool(conv1, 'pool1', 3, 2, padding = 'VALID')
        lrn1 = self.lrn(pool1, 'lrn1')

        print(lrn1.get_shape())
        conv2 = self.conv_layer(lrn1, 'conv2', prune = True, split = 2)
        pool2 = self.maxpool(conv2, 'pool2', 3, 2, padding = 'VALID')
        lrn2 = self.lrn(pool2, 'lrn2')
        # norm2 = self.batch_norm(conv2, 'norm2', train_phase = self.isTrain)
        print(lrn2.get_shape())

        conv3 = self.conv_layer(lrn2, 'conv3', prune = True)
        print(conv3.get_shape())
        # norm3 = self.batch_norm(conv3, 'norm3', train_phase = self.isTrain)
        # pool3 = self.maxpool(norm3, 'pool3', 3, 2)
        conv4 = self.conv_layer(conv3, 'conv4', prune = True, split = 2)
        print(conv4.get_shape())
        # norm4 = self.batch_norm(conv4, 'norm4', train_phase = self.isTrain)
        conv5 = self.conv_layer(conv4, 'conv5', prune = True, split = 2)
        print(conv5.get_shape())
        # norm5 = self.batch_norm(conv5, 'norm5', train_phase = self.isTrain)
        pool5 = self.maxpool(conv5, 'pool5', 3, 2, padding = 'VALID')
        print(pool5.get_shape())
        sys.exit()


        flattened = tf.reshape(pool5, [-1, 6*6*256])
        fc6 = self.fc_layer(flattened, 'fc6', prune = True)
        # norm6 = self.batch_norm(fc6, 'norm6', train_phase = self.isTrain)

        fc7 = self.fc_layer(fc6, 'fc7', prune = True)
        # norm7 = self.batch_norm(fc7, 'norm7', train_phase = self.isTrain)

        fc8 = self.fc_layer(fc7, 'fc8', prune = True, apply_relu = False)
        self.pred = fc8

    def maxpool(self, x, name, filter_size, stride, padding = 'SAME'):
        return tf.nn.max_pool(x, ksize = [1, filter_size, filter_size, 1],
            strides = [1, stride, stride, 1], padding = padding, name = name)

    def lrn(self, x, name, depth_radius = 2, bias = 1.0, alpha = 2e-5, beta = 0.75):
        """
        local response normalization
        ref: https://www.tensorflow.org/api_docs/python/tf/nn/local_response_normalization
        """
        return tf.nn.lrn(x, depth_radius = depth_radius, bias = bias,
            alpha = alpha, beta = beta, name = name)

    def batch_norm(self, x, name, train_phase, data_format = 'NHWC', epsilon = 1e-3):
        """
        TODO: this batch norm has an error
        refs:
        1. https://github.com/ppwwyyxx/tensorpack/blob/a3674b47bfbf0c8b04aaa85d428b109fea0128ca/tensorpack/models/batch_norm.py
        2. https://gist.github.com/tomokishii/0ce3bdac1588b5cca9fa5fbdf6e1c412
        """
        shape = x.get_shape().as_list()
        ndims = len(shape)

        assert ndims in [2,4]
        if ndims == 2:
            data_format = 'NHWC'

        if data_format == 'NCHW':
            n_out = shape[1]
        else:
            n_out = shape[-1]  # channel

        assert n_out is not None, "Input to BatchNorm cannot have unknown channels!"

        with tf.variable_scope(name):
            beta = tf.Variable(tf.constant(0.0, shape = [n_out]),
                name = 'beta', trainable = True)
            gamma = tf.Variable(tf.constant(1.0, shape = [n_out]),
                name = 'gamma', trainable = True)
            axis = list(range(len(x.get_shape())-1))
            batch_mean, batch_var = tf.nn.moments(x, axis, name = 'moments')

            ema = tf.train.ExponentialMovingAverage(decay = 0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)
            mean, var = tf.cond(train_phase,
                mean_var_with_update,
                lambda: (ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, epsilon)
        return normed

    def fc_layer(self, x, name, prune = False, apply_relu = True):
        with tf.variable_scope(name, reuse = True):
            w = tf.get_variable('w')
            b = tf.get_variable('b')
            if prune:
                w = w * self.weights_masks[name]
            ret = tf.nn.xw_plus_b(x,w,b)
            if apply_relu:
                ret = tf.nn.relu(ret)
        return ret

    def conv_layer(self, x, name, padding = 'SAME', stride = 1,
        split = 1, data_format = 'NHWC', prune = False):

        channel_axis = 3 if data_format == 'NHWC' else 1
        with tf.variable_scope(name, reuse = True):
            w = tf.get_variable('w')
            b = tf.get_variable('b')
            if prune:
                w = w * self.weights_masks[name]
            if split == 1:
                conv = tf.nn.conv2d(x, w, [1, stride, stride, 1], padding, data_format=data_format)
                # conv = tf.nn.conv2d(x, w, stride, padding)
            else:
                inputs = tf.split(x, split, channel_axis)
                kernels = tf.split(w, split, 3)
                # outputs = [tf.nn.conv2d(i, k, stride, padding)
                outputs = [tf.nn.conv2d(i, k, [1, stride, stride, 1], padding, data_format=data_format)
                           for i, k in zip(inputs, kernels)]
                conv = tf.concat(outputs, channel_axis)

            # using Relu
            ret = tf.nn.relu(tf.nn.bias_add(conv, b, data_format=data_format), name='output')
        return ret

    def get_placeholders(self):
        self.test = 'hi'
        self.images = tf.placeholder(tf.float32, [None, 224, 224, 3], 'input')
        self.labels = tf.placeholder(tf.int32, [None], 'label')
        self.isTrain = tf.placeholder(tf.bool, name = 'isTrain')

    def _get_variables(self, isload, weights_path = 'DEFAULT'):
        """
        Network architecture definition
        """
        self.keys = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
        kernel_shapes = [
            [11, 11, 3, 96],
            [5, 5, 48, 256],
            [3, 3, 256, 384],
            [3, 3, 192, 384],
            [3, 3, 192, 256],
            [6 * 6 * 256, 4096],
            [4096, 4096],
            [4096, 1000]
        ]
        biase_shape = [
            [96],
            [256],
            [384],
            [384],
            [256],
            [4096],
            [4096],
            [1000]
        ]
        self.weight_shapes = kernel_shapes
        self.biase_shapes = biase_shape
        if isload:
            with open(weights_path+'.npy', 'rb') as f:
                weights, biases = pickle.load(f)
            for i, key in enumerate(self.keys):
                self._init_layerwise_variables(w_shape = kernel_shapes[i],
                    b_shape = biase_shape[i],
                    name = key,
                    w_init = weights[key],
                    b_init = biases[key])
        else:
            for i,key in enumerate(self.keys):
                self._init_layerwise_variables(w_shape = kernel_shapes[i],
                    b_shape = biase_shape[i],
                    name = key)

    def _init_layerwise_variables(self, w_shape, b_shape, name, w_init = None, b_init = None):
        with tf.variable_scope(name):
            if w_init is None:
                w_init = tf.contrib.layers.variance_scaling_initializer()
            else:
                w_init = tf.constant(w_init)
            if b_init is None:
                b_init = tf.constant_initializer()
            else:
                b_init = tf.constant(b_init)
            w = tf.get_variable('w', w_shape, initializer = w_init)
            b = tf.get_variable('b', b_shape, initializer = b_init)

    def _init_weight_masks(self, is_load):
        names = self.keys
        if is_load:
            with open(weights_path+'mask.npy', 'rb') as f:
                self.weights_masks, self.biases_masks= pickle.load(f)
        else:
            self.weights_masks = {}
            self.biases_masks = {}
            for i, key in enumerate(names):
                self.weights_masks[key] = np.ones(self.weight_shapes[i])
                self.biases_masks[key] = np.ones(self.biase_shapes[i])

    def _apply_a_mask(self, mask, var):
        return (var * mask)






#
#
# def save_weights(weights, biases, file_name = 'base.npy'):
#     # print('Saving weights..')
#     keys = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
#     weights_val = {}
#     for op_name in keys:
#         weights_val[op_name] = [weights[op_name].eval(), biases[op_name].eval()]
#     np.save(file_name, weights_val)
#     # print('saved at {}'.format(file_name))
#
# def initialize_weights_mask(first_time_training, mask_dir, file_name):
#     NUM_CHANNELS = 3
#     NUM_CLASSES = 1000
#     if (first_time_training):
#         print('setting initial mask value')
#         weights_mask = {
#             'conv1': np.ones([11, 11, NUM_CHANNELS, 96]),
#             'conv2': np.ones([5, 5, 48, 256]),
#             'conv3': np.ones([3, 3, 256, 384]),
#             'conv4': np.ones([3, 3, 192, 384]),
#             'conv5': np.ones([3, 3, 192, 256]),
#             'fc6': np.ones([6 * 6 * 256, 4096]),
#             'fc7': np.ones([4096, 4096]),
#             'fc8': np.ones([4096, NUM_CLASSES])
#         }
#         biases_mask = {
#             'conv1': np.ones([96]),
#             'conv2': np.ones([256]),
#             'conv3': np.ones([384]),
#             'conv4': np.ones([384]),
#             'conv5': np.ones([256]),
#             'fc6': np.ones([4096]),
#             'fc7': np.ones([4096]),
#             'fc8': np.ones([NUM_CLASSES])
#         }
#
#         # with open(mask_dir + 'maskcov0cov0fc0fc0fc0.pkl', 'wb') as f:
#         #     pickle.dump((weights_mask,biases_mask), f)
#     else:
#         with open(mask_dir + file_name,'rb') as f:
#             (weights_mask, biases_mask) = pickle.load(f)
#     print('weights set')
#     return (weights_mask, biases_mask)
#
# def initialize_variables(new_model = False, weights_path = 'DEFAULT'):
#     NUM_CHANNELS = 3
#     NUM_CLASSES = 1000
#     keys = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']
#     if (new_model):
#         if (weights_path == 'DEFAULT'):
#             WEIGHTS_PATH = 'base.npy'
#         else:
#             WEIGHTS_PATH = weights_path
#     else:
#         if (weights_path == 'DEFAULT'):
#             WEIGHTS_PATH = 'bvlc_alexnet.npy'
#         else:
#             WEIGHTS_PATH = weights_path
#         # call the create function
#     weights = {}
#     biases = {}
#     if (new_model):
#         weights = {
#             'conv1': tf.Variable(tf.truncated_normal([11, 11, NUM_CHANNELS, 96], stddev=5e-2)),
#             'conv2': tf.Variable(tf.truncated_normal([5, 5, 48, 256], stddev=5e-2)),
#             'conv3': tf.Variable(tf.truncated_normal([3, 3, 256, 384], stddev=5e-2)),
#             'conv4': tf.Variable(tf.truncated_normal([3, 3, 192, 384], stddev=5e-2)),
#             'conv5': tf.Variable(tf.truncated_normal([3, 3, 192, 256], stddev=5e-2)),
#             'fc6': tf.Variable(tf.truncated_normal([6 * 6 * 256, 4096], stddev=0.01)),
#             'fc7': tf.Variable(tf.truncated_normal([4096, 4096], stddev=0.01)),
#             'fc8': tf.Variable(tf.truncated_normal([4096, NUM_CLASSES], stddev=0.01))
#         }
#         biases = {
#             'conv1': tf.Variable(tf.constant(0.1, shape=[96])),
#             'conv2': tf.Variable(tf.constant(0.1, shape=[256])),
#             'conv3': tf.Variable(tf.constant(0.1, shape=[384])),
#             'conv4': tf.Variable(tf.constant(0.1, shape=[384])),
#             'conv5': tf.Variable(tf.constant(0.1, shape=[256])),
#             'fc6': tf.Variable(tf.constant(0.1, shape=[4096])),
#             'fc7': tf.Variable(tf.constant(0.1, shape=[4096])),
#             'fc8': tf.Variable(tf.constant(0.1, shape=[NUM_CLASSES]))
#         }
#     else:
#         print('loading from {}'.format(WEIGHTS_PATH))
#         weights_dict = np.load(WEIGHTS_PATH, encoding = 'bytes').item()
#         for key in keys:
#             print(key)
#             for data in weights_dict[key]:
#                 if (len(data.shape) == 1):
#                     biases[key] = tf.Variable(data)
#                 else:
#                     weights[key] = tf.Variable(data)
#                     print(data.shape)
#     return (weights, biases)
#
# def conv_network(images, weights, biases, keep_prob, batch_size = 128):
#     NUM_CLASSES = 1000
#     NUM_CHANNELS = 3
#     # preprocess
#     # mean = tf.constant([104.0069879317889,116.66876761696767,122.678914340678], dtype=tf.float32, shape=[1, 1, 1, 3])
#     mean_RGB = {'B': 104.0069879317889, 'G': 116.66876761696767, 'R': 122.6789143406786}
#     mean = tf.constant([mean_RGB['B'],mean_RGB['G'],mean_RGB['R']], dtype=tf.float32, shape=[1, 1, 1, 3])
#     # mean-subtracted values: [('B', 104.0069879317889), ('G', 116.66876761696767), ('R', 122.6789143406786)]
#     # images = images * 255.0
#     p_images = images - mean
#     # p_images = images
#     # conv1
#     conv1 = conv(p_images, weights['conv1'], 11, 11, 96, 4, 4, padding = 'VALID')
#     pre_activation = tf.nn.bias_add(conv1, biases['conv1'])
#     conv1_act = tf.nn.relu(pre_activation)
#     # conv1 = tf.nn.relu(tf.reshape(pre_activation,conv.get_shape().as_list()))
#     pool1 = max_pool(conv1_act, 3, 3, 2, 2, padding = 'VALID', name = 'pool1')
#     norm1 = lrn(pool1, 2, 2e-05, 0.75, name = 'norm1')
#
#
#     #conv2
#     conv2 = conv(norm1, weights['conv2'], 5, 5, 256, 1, 1, groups = 2)
#     # conv = tf.nn.conv2d(norm1, weights['conv2'], [1, 1, 1, 1], padding='SAME')
#     pre_activation = tf.nn.bias_add(conv2, biases['conv2'])
#     conv2_act = tf.nn.relu(pre_activation)
#     # conv2 = tf.nn.relu(tf.reshape(pre_activation,conv.get_shape().as_list()))
#     pool2 = max_pool(conv2_act, 3, 3, 2, 2, padding = 'VALID', name = 'pool2')
#     norm2 = lrn(pool2, 2, 2e-05, 0.75, name = 'norm2')
#
#     #conv3
#     conv3 = conv(norm2, weights['conv3'], 3, 3, 384, 1, 1)
#     # conv = tf.nn.conv2d(norm2, weights['conv3'], [1, 1, 1, 1], padding='SAME')
#     pre_activation = tf.nn.bias_add(conv3, biases['conv3'])
#     conv3_act = tf.nn.relu(pre_activation)
#     # conv3 = tf.nn.relu(tf.reshape(pre_activation,conv.get_shape().as_list()))
#
#     #conv4
#     conv4 = conv(conv3_act, weights['conv4'], 3, 3, 384, 1, 1, groups = 2)
#     pre_activation = tf.nn.bias_add(conv4, biases['conv4'])
#     conv4_act = tf.nn.relu(pre_activation)
#     # conv4 = tf.nn.relu(tf.reshape(pre_activation,conv.get_shape().as_list()))
#
#     #conv5
#     conv5 = conv(conv4_act, weights['conv5'], 3, 3, 256, 1, 1, groups = 2)
#     # conv = tf.nn.conv2d(conv4, weights['conv5'], [1, 1, 1, 1], padding='SAME')
#     pre_activation = tf.nn.bias_add(conv5, biases['conv5'])
#     conv5 = tf.nn.relu(pre_activation)
#     pool5 = max_pool(conv5, 3, 3, 2, 2, padding = 'VALID', name = 'pool5')
#
#     #fc6
#     flattened = tf.reshape(pool5, [-1, 6*6*256])
#     fc6 = tf.nn.relu(tf.matmul(flattened, weights['fc6']) + biases['fc6'])
#     dropout6 = dropout(fc6, keep_prob)
#
#     # fc7
#     fc7 = tf.nn.relu(tf.matmul(dropout6, weights['fc7']) + biases['fc7'])
#     dropout7 = dropout(fc7, keep_prob)
#
#     fc8 = tf.matmul(dropout7, weights['fc8']) + biases['fc8']
#     return fc8
#
# def conv(x, weights, filter_height, filter_width, num_filters, stride_y, stride_x,
#          padding='SAME', groups=1):
#
#  # Get number of input channels
#     input_channels = int(x.get_shape()[-1])
#
#   # Create lambda function for the convolution
#     convolve = lambda i, k: tf.nn.conv2d(i, k,
#                                        strides = [1, stride_y, stride_x, 1],
#                                        padding = padding)
#
#     if groups == 1:
#         conv = convolve(x, weights)
#     # In the cases of multiple groups, split inputs & weights and
#     else:
#         # Split input and weights and convolve them separately
#         input_groups = tf.split(axis = 3, num_or_size_splits=groups, value=x)
#         weight_groups = tf.split(axis = 3, num_or_size_splits=groups, value=weights)
#         output_groups = [convolve(i, k) for i,k in zip(input_groups, weight_groups)]
#         # Concat the convolved output together again
#         conv = tf.concat(axis = 3, values = output_groups)
#     return conv
#
#
# def max_pool(x, filter_height, filter_width, stride_y, stride_x, name, padding='VALID'):
#     return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1],
#                                                   strides = [1, stride_y, stride_x, 1],
#                                                   padding = padding, name = name)
#
# def lrn(x, radius, alpha, beta, name, bias=1.0):
#     return tf.nn.local_response_normalization(x, depth_radius = radius, alpha = alpha,
#                                           beta = beta, bias = bias, name = name)
#
# def dropout(x, keep_prob):
#     r
