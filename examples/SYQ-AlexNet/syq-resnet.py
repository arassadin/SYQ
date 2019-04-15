#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: syq-alexnet.py
# Author: Julian Faraone (julian.faraone@sydney.edu.au)


# --load /home/stasysp/Envs/shad/SYQ/train_538/model-53125

import cv2
import tensorflow as tf
import argparse
import numpy as np
import multiprocessing
import msgpack
import os, sys

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from quantize import *
from tensorpack.utils.stats import RatioCounter
from tensorpack.tfutils import sessinit, varmanip
from tensorflow.python.keras import backend as K

from tensorpack.models import BatchNormV2, BNReLU, Conv2D, FullyConnected, GlobalAvgPooling, MaxPooling
from tensorpack.tfutils.argscope import argscope, get_arg_scope

TOTAL_BATCH_SIZE = 32
BATCH_SIZE = 4
INP_SIZE = 64
INITIAL = True

BITA = 8
FRAC = 4
PATH = ''


INITIAL = False
weights = None


def activate(x):
    x = tf.nn.relu(x)
    x = tf.clip_by_value(x, 0, 1)
    x = quantize(x, BITA, None)
    return x

def resblock(x, channel, stride):
        def get_stem_full(x):
            return (LinearWrap(x)
                    .Conv2D('c3x3a', channel, 3)
                    .BatchNorm('stembn')
                    .apply(activate)
                    .Conv2D('c3x3b', channel, 3)())
        channel_mismatch = channel != x.get_shape().as_list()[3]
        if stride != 1 or channel_mismatch or 'pool1' in x.name:
            # handling pool1 is to work around an architecture bug in our model
            if stride != 1 or 'pool1' in x.name:
                x = AvgPooling('pool', x, stride, stride)
            x = BatchNorm('bn', x)
            x = activate(x)
            shortcut = Conv2D('shortcut', x, channel, 1)
            stem = get_stem_full(x)
        else:
            shortcut = x
            x = BatchNorm('bn', x)
            x = activate(x)
            stem = get_stem_full(x)
        return shortcut + stem
def group(x, name, channel, nr_block, stride):
        with tf.variable_scope(name + 'blk1'):
            x = resblock(x, channel, stride)
        for i in range(2, nr_block + 1):
            with tf.variable_scope(name + 'blk{}'.format(i)):
                x = resblock(x, channel, 1)
        return x

class Model(ModelDesc):
    def _get_input_vars(self):
        return [InputVar(tf.float32, [None, INP_SIZE, INP_SIZE, 3], 'input'),
                InputVar(tf.int32, [None], 'label')]

    def _build_graph(self, input_vars):
        image, label = input_vars
        image = image / 255.0

        # monkey-patch tf.get_variable to apply fw
        old_get_variable = tf.get_variable
        def new_get_variable(name, shape=None, **kwargs):
            v = old_get_variable(name, shape, **kwargs)

            # don't binarize first and last layer
            if name != 'W' or 'conv0' in v.op.name or 'fct' in v.op.name:
                #print(v)
                return v
            else:
                print("**********  ", name,shape, kwargs)
                tmp = fine_grained_quant(v, args.eta, v.op.name, INITIAL, weights)
                #sess = K.get_session()
                #print(sess.run(tmp))
                #print(tf.keras.backend.eval(tmp))
                #print(tmp)
                return tmp
        tf.get_variable = new_get_variable

        with argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNormV2]):
            # image = LinearWrap(image)
            logits = (LinearWrap(image)
                      # use explicit padding here, because our private training framework has
                      # different padding mechanisms from TensorFlow
                      .tf.pad([[0, 0], [3, 2], [3, 2], [0, 0]])
                      .Conv2D('conv1', 64, 7, stride=2, padding='VALID', use_bias=True)
                      .tf.pad([[0, 0], [1, 1], [1, 1], [0, 0]], 'SYMMETRIC')
                      .MaxPooling('pool1', 3, 2, padding='VALID')
                      .apply(group, 'conv2', 64, 2, 1)
                      .apply(group, 'conv3', 128, 2, 2)
                      .apply(group, 'conv4', 256, 2, 2)
                      .apply(group, 'conv5', 512, 2, 2)
                      .BatchNorm('lastbn')
                      .apply(activate)
                      .GlobalAvgPooling('gap')
                      .tf.multiply(49)  # this is due to a bug in our model design
                      .FullyConnected('fct', 200)())


        tf.get_variable = old_get_variable

        prob = tf.nn.softmax(logits, name='output')

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        wrong = prediction_incorrect(logits, label, 1, name='wrong-top1')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top1'))
        wrong = prediction_incorrect(logits, label, 5, name='wrong-top5')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top5'))

        # weight decay on all W of fc layers
        wd_cost = regularize_cost('fc.*/W', l2_regularizer(5e-6))
        add_moving_summary(cost, wd_cost)

        add_param_summary([('.*/W', ['histogram', 'rms'])])
        self.cost = tf.add_n([cost, wd_cost], name='cost')


def get_data(dataset_name):
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
                return cv2.resize(img, (INP_SIZE, INP_SIZE),
                                  interpolation=cv2.INTER_CUBIC)

        augmentors = [
            imgaug.RandomCrop((INP_SIZE - 10, INP_SIZE - 10)),
            imgaug.RotationAndCropValid(20),
            Resize(),
            imgaug.Flip(horiz=True),
            imgaug.GaussianBlur(),
            imgaug.Brightness(10),
            # imgaug.Contrast(0.1),
            # imgaug.Gamma(),
            # imgaug.Clip(),
            # imgaug.Saturation(0.1),
            # imgaug.Lighting(0.1),
            imgaug.MeanVarianceNormalize(),
            # imgaug.augmenters.PiecewiseAffine(scale=(0.01, 0.05)),
            # imgaug.Contrast((0.8, 1.2)),
            # imgaug.GaussianDeform([(0.2, 0.2), (0.7, 0.2), (0.8, 0.8), (0.5, 0.5), (0.2, 0.5)], (360, 480), 0.2, randrange=20),
            # RandomCropRandomShape(0.3),
            # imgaug.SaltPepperNoise()
            # imgaug.MapImage(lambda x: x - pp_mean_224),
        ]
    else:
        def resize_func(im):
            print('here2')
            h, w = im.shape[:2]
            scale = 256.0 / min(h, w)
            desSize = map(int, (max(INP_SIZE, min(w, scale * w)), \
                                max(INP_SIZE, min(h, scale * h))))
            im = cv2.resize(im, tuple(desSize), interpolation=cv2.INTER_CUBIC)
            return im

        augmentors = [
            imgaug.MeanVarianceNormalize(),
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    if isTrain:
        ds = PrefetchDataZMQ(ds, min(12, multiprocessing.cpu_count()))
    return ds


def get_config(learning_rate, num_epochs, inf_epochs):
    mod = sys.modules['__main__']
    basename = os.path.basename(mod.__file__)
    logdir = os.path.join(PATH, '{}'.format(args.name))

    logger.set_logger_dir(logdir)

    import shutil

    shutil.copy(mod.__file__, logger.LOG_DIR)

    # prepare dataset
    data_train = get_data('train')
    data_test = get_data('val')

    lr = get_scalar_var('learning_rate', learning_rate[0], summary=True)

    lr = tf.Variable(4e-6, trainable=False, name='learning_rate')

    total_epochs = np.arange(1, (num_epochs[-1] + 1))
    do_epochs = np.append(inf_epochs, total_epochs[num_epochs[-2]:])

    return TrainConfig(
        dataset=data_train,
        optimizer=tf.train.AdamOptimizer(lr),
        callbacks=Callbacks([
            StatPrinter(), ModelSaver(),
            # HumanHyperParamSetter('learning_rate'),
            ScheduledHyperParamSetter(
                'learning_rate', zip(num_epochs[:-1], learning_rate[1:])),
            InferenceRunner(data_test,
                            [ScalarStats('cost'),
                             ClassificationError('wrong-top1', 'val-error-top1'),
                             ClassificationError('wrong-top5', 'val-error-top5')], do_epochs)
        ]),
        model=Model(),
        step_per_epoch=3125,  # 100k / batch_size
        max_epoch=num_epochs[-1],
    )


def run_image(model, sess_init, inputs):
    pred_config = PredictConfig(
        model=model,
        session_init=sess_init,
        session_config=get_default_sess_config(0.9),
        input_var_names=['input'],
        output_var_names=['output']
    )
    predict_func = get_predict_func(pred_config)
    meta = dataset.ILSVRCMeta()
    pp_mean = meta.get_per_pixel_mean()
    pp_mean_224 = pp_mean[16:-16, 16:-16, :]
    words = meta.get_synset_words_1000()

    def resize_func(im):
        h, w = im.shape[:2]
        scale = 256.0 / min(h, w)
        desSize = map(int, (max(INP_SIZE, min(w, scale * w)), \
                            max(INP_SIZE, min(h, scale * h))))
        im = cv2.resize(im, tuple(desSize), interpolation=cv2.INTER_CUBIC)
        return im

    transformers = imgaug.AugmentorList([
        imgaug.MapImage(resize_func),
        imgaug.CenterCrop((INP_SIZE, INP_SIZE)),
        imgaug.MapImage(lambda x: x - pp_mean_224),
    ])
    for f in inputs:
        assert os.path.isfile(f)
        img = cv2.imread(f).astype('float32')
        assert img is not None

        img = transformers.augment(img)[np.newaxis, :, :, :]
        outputs = predict_func([img])[0]
        prob = outputs[0]
        ret = prob.argsort()[-10:][::-1]

        names = [words[i] for i in ret]
        print(f + ":")
        print(list(zip(names, prob[ret])))


def eval_on_ILSVRC12(model_path, data_dir, ds_type):
    ds = get_data(ds_type)
    pred_config = PredictConfig(
        model=Model(),
        session_init=get_model_loader(model_path),
        input_names=['input', 'label'],
        output_names=['wrong-top1', 'wrong-top5']
    )
    pred = SimpleDatasetPredictor(pred_config, ds)
    acc1, acc5 = RatioCounter(), RatioCounter()
    for o in pred.get_result():
        batch_size = o[0].shape[0]
        acc1.feed(o[0].sum(), batch_size)
        acc5.feed(o[1].sum(), batch_size)
    print("Top1 Error: {}".format(acc1.ratio))
    print("Top5 Error: {}".format(acc5.ratio))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='the physical ids of GPUs to use')
    parser.add_argument('--load', help='load a checkpoint, or a npy (given as the pretrained model)')
    parser.add_argument('--data', help='ILSVRC dataset dir', default='/home/stasysp/Envs/shad/SYQ/tiny-imagenet-200')
                        #default='/home/stasysp/Envs/Datasets/ImageNet')
    parser.add_argument('--run', help='run on a list of images with the pretrained model', nargs='*')
    parser.add_argument('--eta', type=float, default=0)
    parser.add_argument('--learning-rate', type=float, nargs='+', metavar='LR', default=[1e-3, 2e-5, 4e-6],
            help='Learning rates to use during training, first value is the initial learning rate (default: %(default)s). Must have the same number of args as --num-epochs')
    parser.add_argument('--num-epochs', type=int, nargs='+', metavar='E', default=[100000, 150, 100],
            help='Epochs to change the learning rate, last value is the maximum number of epochs (default: %(default)s). Must have the same number of args as --learning-rate')
    parser.add_argument('--inf-epochs', type=int, nargs='+', metavar='I', default=list(np.arange(1,121)))
    parser.add_argument('--eval', type=str, default=None, choices=['val', 'test'],
            help='evaluate the model on the test of validation set')
    parser.add_argument('--name', default='resnet_bin0eta')

    args = parser.parse_args()

    if args.eval != None:
        eval_on_ILSVRC12(args.load, args.data, args.eval)
        sys.exit()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.run:
        assert args.load.endswith('.npy')
        run_image(Model(), ParamRestore(np.load(args.load, encoding='latin1').item()), args.run)
        sys.exit()

    assert args.gpu is not None, "Need to specify a list of gpu for training!"
    NR_GPU = len(args.gpu.split(','))
    BATCH_SIZE = TOTAL_BATCH_SIZE // NR_GPU

    assert len(args.num_epochs) == len(args.learning_rate)
    config = get_config(args.learning_rate, args.num_epochs, args.inf_epochs)

    if args.load:
        if args.load.endswith('.npy'):
            config.session_init = ParamRestore(np.load(args.load, encoding='latin1').item())
        else:
            config.session_init = SaverRestore(args.load)

    if args.gpu:
        config.nr_tower = len(args.gpu.split(','))

    SyncMultiGPUTrainer(config).train()
    #varmanip.dump_session_params('alexnet_v01.npy')
