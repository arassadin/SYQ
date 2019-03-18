#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: ilsvrc.py
import os
import cv2
import numpy as np
from six.moves import range
import xml.etree.ElementTree as ET

from ...utils import logger, get_rng, get_dataset_path
from ...utils.fs import mkdir_p
from ...utils.timer import timed_operation
from ..base import RNGDataFlow

__all__ = ['TinyMeta', 'Tiny']

class TinyMeta(object):
    """
    Some metadata for Tiny dataset.
    """
    def __init__(self, dir=None):
        if dir is None:
            dir = get_dataset_path('tiny_metadata')#'ilsvrc_metadata')
        self.dir = dir
        mkdir_p(self.dir)

    def get_synsets(self):
        """
        :returns a dict of {cls_number: synset_id}
        """
        fname = os.path.join(self.dir, 'wnids.txt')
        assert os.path.isfile(fname)
        lines = [x.strip() for x in open(fname).readlines()]
        return dict(enumerate(lines))

    def get_image_list(self, name):
        """
        :param name: 'train' or 'val' or 'test'
        :returns: list of (image filename, cls)
        """
        assert name in ['train', 'val', 'test']
        fname = os.path.join(self.dir, name + '.txt')
        assert os.path.isfile(fname)
        with open(fname) as f:
            ret = []
            for line in f.readlines():
                name, cls = line.strip().split()
                ret.append((name, int(cls)))
        assert len(ret)
        return ret

    def get_per_pixel_mean(self, size=None):
        """
        :param size: return image size in [h, w]. default to (256, 256)
        :returns: per-pixel mean as an array of shape (h, w, 3) in range [0, 255]
        """

        return np.zeros((*size, 3))

class Tiny(RNGDataFlow):
    def __init__(self, dir, name, meta_dir=None, shuffle=True,
            dir_structure='not original'):
        """
        :param dir: A directory containing a subdir named `name`, where the
            original ILSVRC12_`name`.tar gets decompressed.
        :param name: 'train' or 'val' or 'test'
        :param dir_structure: The dir structure of 'val' and 'test'.
            If is 'original' then keep the original decompressed directory with list
            of image files (as below). If set to 'train', use the the same
            directory structure as 'train/', with class name as subdirectories.

        When `dir_structure=='original'`, `dir` should have the following structure:

        .. code-block:: none

            dir/
              train/
                n02134418/
                  n02134418_198.JPEG
                  ...
                ...
              val/
                ILSVRC2012_val_00000001.JPEG
                ...
              test/
                ILSVRC2012_test_00000001.JPEG
                ...
              bbox/
                n02134418/
                  n02134418_198.xml
                  ...
                ...

        After decompress ILSVRC12_img_train.tar, you can use the following
        command to build the above structure for `train/`:

        .. code-block:: none

            tar xvf ILSVRC12_img_train.tar -C train && cd train
            find -type f -name '*.tar' | parallel -P 10 'echo {} && mkdir -p {/.} && tar xf {} -C {/.}'
            Or:
            for i in *.tar; do dir=${i%.tar}; echo $dir; mkdir -p $dir; tar xf $i -C $dir; done

        """
        assert name in ['train', 'test', 'val']
        self.full_dir = os.path.join(dir, name)
        self.name = name
        assert os.path.isdir(self.full_dir), self.full_dir
        self.shuffle = shuffle
        meta = TinyMeta(meta_dir)
        self.imglist = meta.get_image_list(name)
        self.dir_structure = dir_structure
        self.synset = meta.get_synsets()

    def size(self):
        return len(self.imglist)

    def get_data(self):
        """
        Produce original images of shape [h, w, 3(BGR)] and label
        """
        idxs = np.arange(len(self.imglist))
        add_label_to_fname = (self.name != 'train' and self.dir_structure != 'original' and self.name != 'val')
        print('#######', add_label_to_fname)
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            fname, label = self.imglist[k]
            if add_label_to_fname:
                fname = os.path.join(self.full_dir, self.synset[label], fname)
            else:
                fname = os.path.join(self.full_dir, fname)
            im = cv2.imread(fname.strip(), cv2.IMREAD_COLOR)
            assert im is not None, fname
            if im.ndim == 2:
                im = np.expand_dims(im, 2).repeat(3, 2)
            yield [im, label]
