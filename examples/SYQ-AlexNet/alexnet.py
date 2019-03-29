import cv2
import tensorflow as tf
import argparse
import numpy as np
import os, sys

sys.path.append('/home/alexandr/develop/SYQ')

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.models.batch_norm import BatchNorm


BATCH_SIZE = 64
INP_SIZE = 64
PATH = ''


class Model(ModelDesc):
    def _get_input_vars(self):
        return [InputVar(tf.float32, [None, INP_SIZE, INP_SIZE, 3], 'input'),
                InputVar(tf.int32, [None], 'label') ]

    def _build_graph(self, input_vars):
        image, label = input_vars
        # image = image / 255.0

        gauss_init = tf.random_normal_initializer(stddev=0.01)

        def activate(x):
            return tf.nn.relu(x)

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

                .FullyConnected('fc0', 4096, W_init=gauss_init)
                .BatchNorm('bnfc0')
                .apply(activate)

                .FullyConnected('fc1', 4096, W_init=gauss_init)
                .BatchNorm('bnfc1')
                .apply(activate)
                .FullyConnected('fct', 200, use_bias=True)()
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
    ds = dataset.Tiny(args.data, dataset_name, shuffle=isTrain)

    augmentors = []

    if isTrain:
        augmentors += [
            imgaug.Lighting(0.1,
                            eigval=np.asarray(
                                [0.2175, 0.0188, 0.0045][::-1]) * 255.0,
                            eigvec=np.array(
                                [[-0.5675, 0.7192, 0.4009],
                                 [-0.5808, -0.0045, -0.8140],
                                 [-0.5836, -0.6948, 0.4203]],
                                dtype='float32')[::-1, ::-1]),
            imgaug.Flip(horiz=True)
        ]

    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)

    if isTrain:
        ds = PrefetchDataZMQ(ds, 5)

    return ds

def get_config(lrs, epochs, epochs_inf, epoch_n):
    logdir = os.path.join(PATH, 'logdir')
    logger.set_logger_dir(logdir)

    # mod = sys.modules['__main__']
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
    parser.add_argument('--data', help='ILSVRC dataset dir', default='/home/alexandr/datasets/tiny-imagenet-200')
    parser.add_argument('--lrs', type=float, nargs='+', metavar='LR', default=[1e-3],
                        help='Learning rates to use during training, first value is the initial learning rate')
    parser.add_argument('--epochs', type=int, nargs='+', metavar='E', default=[0],
                        help='Epochs to change the learning rate, last value is the maximum number of epochs')
    parser.add_argument('--epochs-inf', type=int, nargs='+', metavar='I', default=list(np.arange(500)))
    parser.add_argument('--epoch_n', type=int, default=500)
    args = parser.parse_args()

    assert len(args.epochs) == len(args.lrs)

    config = get_config(args.lrs, args.epochs, args.epochs_inf, args.epoch_n)
    # config.nr_tower = 1

    # SyncMultiGPUTrainer(config).train()
    SimpleTrainer(config).train()
