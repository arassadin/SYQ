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
#from quantize import *
from tensorpack.utils.stats import RatioCounter

TOTAL_BATCH_SIZE = 64
BATCH_SIZE = 64
INP_SIZE = 64
INITIAL = True

BITA = 8
FRAC = 4
PATH = ''
use_local_stat = True

PATH_float = '../floatingpoint_alexnet.npy'

if INITIAL:
    d = np.load(PATH_float, encoding='latin1').item()
    weights = {}
    #calculate initialization for scaling coefficients
    for i in d.keys():
        if '/W:' in i and 'conv' in i:
            mean = np.mean(np.absolute(d[i]), axis = (2,3))
            weights[i] = mean
        elif '/W:' in i and 'fc' in i:
            mean = np.mean(np.absolute(d[i]))
            weights[i] = mean
else:
    weights = None

from tensorpack.models import BatchNormV2, BNReLU, Conv2D, FullyConnected, GlobalAvgPooling, MaxPooling
from tensorpack.tfutils.argscope import argscope, get_arg_scope

def activate(x):
    x = tf.nn.relu(x) #BNReLU(x) #
    return x   

def resnet_shortcut(l, n_out, stride, activation=tf.identity):
    global use_local_stat
    #data_format = get_arg_scope()['Conv2D']['data_format']
    n_in = l.shape[3] #l.get_shape().as_list()[1 if data_format in ['NCHW', 'channels_first'] else 3]
    #print(n_in)
    if n_in != n_out:   # change dimension when channel is not the same   
        l = Conv2D('convshortcut', l, n_out, 1, stride=stride)#, activation=activation)
        l = BatchNormV2('convshortcut', l, use_local_stat=use_local_stat)
        return activate(l)
    else:
        return l


def get_bn(x, zero_init=False):
    """
    Zero init gamma is good for resnet. See https://arxiv.org/abs/1706.02677.
    """
    global use_local_stat
    if zero_init:
        #return lambda x, name=None: BatchNorm('bn', x, gamma_initializer=tf.zeros_initializer())
        return BatchNormV2('bn_z', x, use_local_stat=use_local_stat) #, gamma_initializer=tf.zeros_initializer())
    else:
        #return lambda x, name=None: BatchNorm('bn', x)
        return BatchNormV2('bn', x, use_local_stat=use_local_stat)


def resnet_bottleneck(l, ch_out, stride, stride_first=False):
    """
    stride_first: original resnet put stride on first conv. fb.resnet.torch put stride on second conv.
    """
    global use_local_stat
    shortcut = l
    l = Conv2D('conv1', l, ch_out, 1, stride=stride if stride_first else 1)#.apply(BNReLU)
    l = BatchNormV2('bn1', l, use_local_stat=use_local_stat)
    l = activate(l)
    l = Conv2D('conv2', l, ch_out, 3, stride=1 if stride_first else stride)#.apply(BNReLU)
    l = BatchNormV2('bn2', l, use_local_stat=use_local_stat)
    l = activate(l)
    l = get_bn(Conv2D('conv3', l, ch_out * 4, 1), zero_init=True)

    out = l + get_bn(resnet_shortcut(shortcut, ch_out * 4, stride))
    return tf.nn.relu(out)

def resnet_group(name, l, block_func, features, count, stride):
    with tf.variable_scope(name):
        for i in range(0, count):
            with tf.variable_scope('block{}'.format(i)):
                l = block_func(l, features, stride if i == 0 else 1)
    return l


def resnet_backbone(image, num_blocks, group_func, block_func):
    with argscope(Conv2D, use_bias=False): #kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out')):
        # Note that this pads the image by [2, 3] instead of [3, 2].
        # Similar things happen in later stride=2 layers as well.
        l = activate(Conv2D('conv0', image, 64, 7, stride=2))#, activation=BNReLU)
        l = MaxPooling('pool0', l, 3, stride=2, padding='SAME')
        l = group_func('group0', l, block_func, 64, num_blocks[0], 1)
        l = group_func('group1', l, block_func, 128, num_blocks[1], 2)
        l = group_func('group2', l, block_func, 256, num_blocks[2], 2)
        l = group_func('group3', l, block_func, 512, num_blocks[3], 2)
        l = GlobalAvgPooling('gap', l)
        logits = FullyConnected('linear', l, 1000)#, initializer=tf.random_normal_initializer(stddev=0.01))
    return logits

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
                InputVar(tf.int32, [None], 'label') ]


    def _build_graph(self, input_vars):
        image, label = input_vars
        image = image / 255.0
        pass

#         def activate(x):
#             x = BNReLU(x) #tf.nn.relu(x) #BNReLU(x) #
#             return x
        
        with argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNormV2]):
            #image = LinearWrap(image)
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

        #tf.get_variable = old_get_variable

        prob = tf.nn.softmax(logits, name='output')

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        self.cost = tf.reduce_mean(cost, name='cost') #cost #tf.Variable(cost, name='cost') #
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        wrong = prediction_incorrect(logits, label, 1, name='wrong-top1')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top1'))
        wrong = prediction_incorrect(logits, label, 5, name='wrong-top5')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top5'))

        # weight decay on all W of fc layers
        #wd_cost = regularize_cost('linear.*/W', l2_regularizer(5e-6))
        #add_moving_summary(cost, wd_cost) #########
        
        add_param_summary([('.*/W', ['histogram', 'rms'])])
        #print(cost)
        #print(wd_cost)
     

def get_data(dataset_name):
    isTrain = dataset_name == 'train'
    ds = dataset.ILSVRC12(args.data, dataset_name, shuffle=isTrain)

    meta = dataset.ILSVRCMeta()
    pp_mean = meta.get_per_pixel_mean()
    pp_mean_224 = pp_mean[16:-16,16:-16,:]

    if isTrain:
        class Resize(imgaug.ImageAugmentor):
            def __init__(self):
                self._init(locals())
            def _augment(self, img, _):
                return  cv2.resize(img, (INP_SIZE, INP_SIZE),
                      interpolation=cv2.INTER_CUBIC)

        augmentors = [
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
            imgaug.MeanVarianceNormalize(),
            #imgaug.augmenters.PiecewiseAffine(scale=(0.01, 0.05)),
            #imgaug.Contrast((0.8, 1.2)),
            #imgaug.GaussianDeform([(0.2, 0.2), (0.7, 0.2), (0.8, 0.8), (0.5, 0.5), (0.2, 0.5)], (360, 480), 0.2, randrange=20),
            # RandomCropRandomShape(0.3),
            #imgaug.SaltPepperNoise()
            #imgaug.MapImage(lambda x: x - pp_mean_224),
        ]
    else:
        augmentors = [
            imgaug.MeanVarianceNormalize(),
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    if isTrain:
        ds = PrefetchDataZMQ(ds, min(12, multiprocessing.cpu_count()))
    return ds

def optimizer(lr):
    #tf.summary.scalar('learning_rate-summary', lr)
    return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)

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


    total_epochs = np.arange(1, (num_epochs[-1] + 1))
    do_epochs = np.append(inf_epochs, total_epochs[num_epochs[-2]:])

    return TrainConfig(
        dataset=data_train,
        optimizer=optimizer(lr),
        callbacks=Callbacks([
            StatPrinter(), ModelSaver(),
            #HumanHyperParamSetter('learning_rate'),
            ScheduledHyperParamSetter(
                'learning_rate', zip(num_epochs[:-1], learning_rate[1:])),
            InferenceRunner(data_test,
                [ScalarStats('cost'),
                 ClassificationError('wrong-top1', 'val-error-top1'),
                 ClassificationError('wrong-top5', 'val-error-top5')], do_epochs)
        ]),
        model=Model(),
        step_per_epoch = 100000 // BATCH_SIZE, # 100k / batch_size
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
    pp_mean_224 = pp_mean[16:-16,16:-16,:]
    words = meta.get_synset_words_1000()

    def resize_func(im):
        h, w = im.shape[:2]
        scale = 256.0 / min(h, w)
        desSize = map(int, (max(INP_SIZE, min(w, scale * w)),\
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

        img = transformers.augment(img)[np.newaxis, :,:,:]
        outputs = predict_func([img])[0]
        prob = outputs[0]
        ret = prob.argsort()[-10:][::-1]

        names = [words[i] for i in ret]
        print(f + ":")
        print(list(zip(names, prob[ret])))

def eval_on_ILSVRC12(model_path, data_dir, ds_type):
    global use_local_stat
    use_local_stat = False
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
    use_local_stat = True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='the physical ids of GPUs to use')
    parser.add_argument('--load', help='load a checkpoint, or a npy (given as the pretrained model)')
    parser.add_argument('--data', help='ILSVRC dataset dir', default='/home/stasysp/Envs/shad/SYQ/tiny-imagenet-200')
                        #default='/home/stasysp/Envs/Datasets/ImageNet')
    parser.add_argument('--run', help='run on a list of images with the pretrained model', nargs='*')
    parser.add_argument('--eta', type=float, default=0)
    parser.add_argument('--learning-rate', type=float, nargs='+', metavar='LR', default=[1e-2, 1e-2, 1e-2],
            help='Learning rates to use during training, first value is the initial learning rate (default: %(default)s). Must have the same number of args as --num-epochs')
    parser.add_argument('--num-epochs', type=int, nargs='+', metavar='E', default=[100000, 150, 200],
            help='Epochs to change the learning rate, last value is the maximum number of epochs (default: %(default)s). Must have the same number of args as --learning-rate')
    parser.add_argument('--inf-epochs', type=int, nargs='+', metavar='I', default=list(np.arange(1,121)))
    parser.add_argument('--eval', type=str, default=None, choices=['val', 'test'],
            help='evaluate the model on the test of validation set')
    parser.add_argument('--name', default='resnet50')

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
