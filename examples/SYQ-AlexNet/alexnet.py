import cv2
import tensorflow as tf
import argparse
import numpy as np
import os, sys

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *


BATCH_SIZE = 32
INP_SIZE = 64
PATH = ''


class Model(ModelDesc):
    def _get_input_vars(self):
        return [InputVar(tf.float32, [None, INP_SIZE, INP_SIZE, 3], 'input'),
                InputVar(tf.int32, [None], 'label') ]

    def _build_graph(self, input_vars):
        image, label = input_vars
        image = image / 255.0

        def activate(x):
            x = tf.nn.relu(x)
            return x

        with argscope(BatchNorm, decay=0.9, epsilon=1e-4), argscope([Conv2D, FullyConnected], use_bias=False, nl=tf.identity):
            logits = (
                LinearWrap(image)
                .Conv2D('conv0', 96, 3, stride=2, padding='VALID')
                .apply(activate)
                .Conv2D('conv1', 256, 3, padding='SAME', split=2)
                .BatchNorm('bn1')
                .MaxPooling('pool1', 3, 2, padding='SAME')
                .apply(activate)

                .Conv2D('conv2', 384, 3)
                .BatchNorm('bn2')
                .MaxPooling('pool2', 3, 1, padding='SAME')
                .apply(activate)

                .Conv2D('conv3', 384, 3, split=2)
                .BatchNorm('bn3')
                .apply(activate)

                .Conv2D('conv4', 256, 3, split=2)
                .BatchNorm('bn4')
                .MaxPooling('pool4', 5, 2, padding='VALID')
                .apply(activate)

                .FullyConnected('fc0', 4096)
                .BatchNorm('bnfc0')
                .apply(activate)

                .FullyConnected('fc1', 4096)
                .BatchNorm('bnfc1')
                .apply(activate)
                .FullyConnected('fct', 1000, use_bias=True)()
            )

        prob = tf.nn.softmax(logits, name='output')

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        wrong = prediction_incorrect(logits, label, 1, name='wrong-top1')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top1'))
        wrong = prediction_incorrect(logits, label, 5, name='wrong-top5')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top5'))

        wd_cost = regularize_cost('fc.*/W', l2_regularizer(5e-6))
        add_moving_summary(cost, wd_cost)

        add_param_summary([('.*/W', ['histogram', 'rms'])])
        self.cost = tf.add_n([cost, wd_cost], name='cost')

def get_data(dataset_name):
    isTrain = dataset_name == 'train'
    ds = dataset.ILSVRC12(args.data, dataset_name, shuffle=isTrain)

    # meta = dataset.ILSVRCMeta()
    # pp_mean = meta.get_per_pixel_mean()
    # pp_mean_224 = pp_mean[16:-16, 16:-16,:]

    augmentors = [imgaug.MeanVarianceNormalize()]
    if isTrain:
        class Resize(imgaug.ImageAugmentor):

            def __init__(self):
                self._init(locals())

            def _augment(self, img, _):
                return  cv2.resize(img, (INP_SIZE, INP_SIZE), interpolation=cv2.INTER_CUBIC)

        augmentors += [
            imgaug.RandomCrop((INP_SIZE - 10, INP_SIZE - 10)),
            imgaug.RotationAndCropValid(20),
            Resize(),
            imgaug.Flip(horiz=True),
            imgaug.GaussianBlur(),
            imgaug.Brightness(10),
            #imgaug.Contrast(0.1),
            #imgaug.Gamma(),
            #imgaug.Clip(),
            #imgaug.Saturation(0.1),
            #imgaug.Lighting(0.1),
            #imgaug.augmenters.PiecewiseAffine(scale=(0.01, 0.05)),
            #imgaug.Contrast((0.8, 1.2)),
            #imgaug.GaussianDeform([(0.2, 0.2), (0.7, 0.2), (0.8, 0.8), (0.5, 0.5), (0.2, 0.5)], (360, 480), 0.2, randrange=20),
            # RandomCropRandomShape(0.3),
            #imgaug.SaltPepperNoise()
            #imgaug.MapImage(lambda x: x - pp_mean_224),
        ]

    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)

    if isTrain:
        ds = PrefetchDataZMQ(ds, 5)

    return ds

def get_config(lrs, epochs, epochs_inf, epoch_n):
    mod = sys.modules['__main__']

    logdir = os.path.join(PATH, '{}'.format(args.name))
    logger.set_logger_dir(logdir)

    # import shutil
    # shutil.copy(mod.__file__, logger.LOG_DIR)

    data_train = get_data('train')
    data_test = get_data('val')

    lr = get_scalar_var('learning_rate', lrs[0], summary=True)

    return TrainConfig(
        dataset=data_train,
        optimizer=tf.train.AdamOptimizer(lrs[0]),
        callbacks=Callbacks([
            StatPrinter(),
            # ModelSaver(),
            #HumanHyperParamSetter('learning_rate'),
            # ScheduledHyperParamSetter(
            #     'learning_rate', zip(epochs[:-1], lrs[1:])
            # ),
            InferenceRunner(
                data_test,
                [ScalarStats('cost'),
                 ClassificationError('wrong-top1', 'val-error-top1'),
                 ClassificationError('wrong-top5', 'val-error-top5')], 
                epochs_inf
            )
        ]),
        model=Model(),
        step_per_epoch=3125,
        max_epoch=epoch_n
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='ILSVRC dataset dir', default='/home/stasysp/Envs/shad/SYQ/tiny-imagenet-200')
    parser.add_argument('--lrs', type=float, nargs='+', metavar='LR', default=[1e-3],
                        help='Learning rates to use during training, first value is the initial learning rate')
    parser.add_argument('--epochs', type=int, nargs='+', metavar='E', default=[0],
                        help='Epochs to change the learning rate, last value is the maximum number of epochs')
    parser.add_argument('--epochs-inf', type=int, nargs='+', metavar='I', default=list(np.arange(1,121)))
    parser.add_argument('--epoch_n', type=int, default=500)
    args = parser.parse_args()

    assert len(args.epochs) == len(args.lrs)

    config = get_config(args.lrs, args.epochs, args.epochs_inf, args.epoch_n)
    config.nr_tower = 0

    SyncMultiGPUTrainer(config).train()
