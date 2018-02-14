# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('bash', '', 'rm -rf data\nmkdir data\naws s3 sync s3://ps-jiaan01-artifacts/input/data data/')


# In[2]:


get_ipython().run_cell_magic('bash', '', 'rm -rf model\nmkdir model\naws s3 cp s3://ps-jiaan01-artifacts/input/model/vgg16_reduced-symbol.json model/\naws s3 cp s3://ps-jiaan01-artifacts/input/model/vgg16_reduced-0001.params model/')


# In[5]:


get_ipython().run_cell_magic('bash', '', 'rm -rf log\nmkdir log\nrm -rf out\nmkdir out')


# In[6]:


import os, logging, re, cv2
import mxnet as mx
import numpy as np
from mxnet import image


# In[7]:


class DetRecordIter(mx.io.DataIter):
    """
    The new detection iterator wrapper for mx.io.ImageDetRecordIter which is
    written in C++, it takes record file as input and runs faster.
    Supports various augment operations for object detection.

    Parameters:
    -----------
    path_imgrec : str
        path to the record file
    path_imglist : str
        path to the list file to replace the labels in record
    batch_size : int
        batch size
    data_shape : tuple
        (3, height, width)
    label_width : int
        specify the label width, use -1 for variable length
    label_pad_width : int
        labels must have same shape in batches, use -1 for automatic estimation
        in each record, otherwise force padding to width in case you want t
        rain/validation to match the same width
    label_pad_value : float
        label padding value
    resize_mode : str
        force - resize to data_shape regardless of aspect ratio
        fit - try fit to data_shape preserving aspect ratio
        shrink - shrink to data_shape only, preserving aspect ratio
    mean_pixels : list or tuple
        mean values for red/green/blue
    kwargs : dict
        see mx.io.ImageDetRecordIter

    Returns:
    ----------

    """
    def __init__(self, path_imgrec, batch_size, data_shape, path_imglist="",
                 label_width=-1, label_pad_width=-1, label_pad_value=-1,
                 resize_mode='force',  mean_pixels=[123.68, 116.779, 103.939],
                 **kwargs):
        super(DetRecordIter, self).__init__()
        self.rec = mx.io.ImageDetRecordIter(
            path_imgrec     = path_imgrec,
            path_imglist    = path_imglist,
            label_width     = label_width,
            label_pad_width = label_pad_width,
            label_pad_value = label_pad_value,
            batch_size      = batch_size,
            data_shape      = data_shape,
            mean_r          = mean_pixels[0],
            mean_g          = mean_pixels[1],
            mean_b          = mean_pixels[2],
            resize_mode     = resize_mode,
            **kwargs)

        self.provide_label = None
        self._get_batch()
        if not self.provide_label:
            raise RuntimeError("Invalid ImageDetRecordIter: " + path_imgrec)
        self.reset()

    @property
    def provide_data(self):
        return self.rec.provide_data

    def reset(self):
        self.rec.reset()

    def iter_next(self):
        return self._get_batch()

    def next(self):
        if self.iter_next():
            return self._batch
        else:
            raise StopIteration

    def _get_batch(self):
        self._batch = self.rec.next()
        if not self._batch:
            return False

        if self.provide_label is None:
            # estimate the label shape for the first batch, always reshape to n*5
            first_label = self._batch.label[0][0].asnumpy()
            self.batch_size = self._batch.label[0].shape[0]
            self.label_header_width = int(first_label[4])
            self.label_object_width = int(first_label[5])
            assert self.label_object_width >= 5, "object width must >=5"
            self.label_start = 4 + self.label_header_width
            self.max_objects = (first_label.size - self.label_start) // self.label_object_width
            self.label_shape = (self.batch_size, self.max_objects, self.label_object_width)
            self.label_end = self.label_start + self.max_objects * self.label_object_width
            self.provide_label = [('label', self.label_shape)]

        # modify label
        label = self._batch.label[0].asnumpy()
        label = label[:, self.label_start:self.label_end].reshape(
            (self.batch_size, self.max_objects, self.label_object_width))
        self._batch.label = [mx.nd.array(label)]
        return True


# In[8]:


class MultiBoxMetric(mx.metric.EvalMetric):
    """Calculate metrics for Multibox training """
    def __init__(self, eps=1e-8):
        super(MultiBoxMetric, self).__init__('MultiBox')
        self.eps = eps
        self.num = 2
        self.name = ['CrossEntropy', 'SmoothL1']
        self.reset()

    def reset(self):
        """
        override reset behavior
        """
        if getattr(self, 'num', None) is None:
            self.num_inst = 0
            self.sum_metric = 0.0
        else:
            self.num_inst = [0] * self.num
            self.sum_metric = [0.0] * self.num

    def update(self, labels, preds):
        """
        Implementation of updating metrics
        """
        # get generated multi label from network
        cls_prob = preds[0].asnumpy()
        loc_loss = preds[1].asnumpy()
        cls_label = preds[2].asnumpy()
        valid_count = np.sum(cls_label >= 0)
        # overall accuracy & object accuracy
        label = cls_label.flatten()
        mask = np.where(label >= 0)[0]
        indices = np.int64(label[mask])
        prob = cls_prob.transpose((0, 2, 1)).reshape((-1, cls_prob.shape[1]))
        prob = prob[mask, indices]
        self.sum_metric[0] += (-np.log(prob + self.eps)).sum()
        self.num_inst[0] += valid_count
        # smoothl1loss
        self.sum_metric[1] += np.sum(loc_loss)
        self.num_inst[1] += valid_count

    def get(self):
        """Get the current evaluation result.
        Override the default behavior

        Returns
        -------
        name : str
           Name of the metric.
        value : float
           Value of the evaluation.
        """
        if self.num is None:
            if self.num_inst == 0:
                return (self.name, float('nan'))
            else:
                return (self.name, self.sum_metric / self.num_inst)
        else:
            names = ['%s'%(self.name[i]) for i in range(self.num)]
            values = [x / y if y != 0 else float('nan')                 for x, y in zip(self.sum_metric, self.num_inst)]
            return (names, values)


# In[9]:


class MApMetric(mx.metric.EvalMetric):
    """
    Calculate mean AP for object detection task

    Parameters:
    ---------
    ovp_thresh : float
        overlap threshold for TP
    use_difficult : boolean
        use difficult ground-truths if applicable, otherwise just ignore
    class_names : list of str
        optional, if provided, will print out AP for each class
    pred_idx : int
        prediction index in network output list
    """
    def __init__(self, ovp_thresh=0.5, use_difficult=False, class_names=None, pred_idx=0):
        super(MApMetric, self).__init__('mAP')
        if class_names is None:
            self.num = None
        else:
            assert isinstance(class_names, (list, tuple))
            for name in class_names:
                assert isinstance(name, str), "must provide names as str"
            num = len(class_names)
            self.name = class_names + ['mAP']
            self.num = num + 1
        self.reset()
        self.ovp_thresh = ovp_thresh
        self.use_difficult = use_difficult
        self.class_names = class_names
        self.pred_idx = int(pred_idx)

    def reset(self):
        """Clear the internal statistics to initial state."""
        if getattr(self, 'num', None) is None:
            self.num_inst = 0
            self.sum_metric = 0.0
        else:
            self.num_inst = [0] * self.num
            self.sum_metric = [0.0] * self.num
        self.records = dict()
        self.counts = dict()

    def get(self):
        """Get the current evaluation result.

        Returns
        -------
        name : str
           Name of the metric.
        value : float
           Value of the evaluation.
        """
        self._update()  # update metric at this time
        if self.num is None:
            if self.num_inst == 0:
                return (self.name, float('nan'))
            else:
                return (self.name, self.sum_metric / self.num_inst)
        else:
            names = ['%s'%(self.name[i]) for i in range(self.num)]
            values = [x / y if y != 0 else float('nan')                 for x, y in zip(self.sum_metric, self.num_inst)]
            return (names, values)

    def update(self, labels, preds):
        """
        Update internal records. This function now only update internal buffer,
        sum_metric and num_inst are updated in _update() function instead when
        get() is called to return results.

        Params:
        ----------
        labels: mx.nd.array (n * 6) or (n * 5), difficult column is optional
            2-d array of ground-truths, n objects(id-xmin-ymin-xmax-ymax-[difficult])
        preds: mx.nd.array (m * 6)
            2-d array of detections, m objects(id-score-xmin-ymin-xmax-ymax)
        """
        def iou(x, ys):
            """
            Calculate intersection-over-union overlap
            Params:
            ----------
            x : numpy.array
                single box [xmin, ymin ,xmax, ymax]
            ys : numpy.array
                multiple box [[xmin, ymin, xmax, ymax], [...], ]
            Returns:
            -----------
            numpy.array
                [iou1, iou2, ...], size == ys.shape[0]
            """
            ixmin = np.maximum(ys[:, 0], x[0])
            iymin = np.maximum(ys[:, 1], x[1])
            ixmax = np.minimum(ys[:, 2], x[2])
            iymax = np.minimum(ys[:, 3], x[3])
            iw = np.maximum(ixmax - ixmin, 0.)
            ih = np.maximum(iymax - iymin, 0.)
            inters = iw * ih
            uni = (x[2] - x[0]) * (x[3] - x[1]) + (ys[:, 2] - ys[:, 0]) *                 (ys[:, 3] - ys[:, 1]) - inters
            ious = inters / uni
            ious[uni < 1e-12] = 0  # in case bad boxes
            return ious

        # independant execution for each image
        for i in range(labels[0].shape[0]):
            # get as numpy arrays
            label = labels[0][i].asnumpy()
            if np.sum(label[:, 0] >= 0) < 1:
                continue
            pred = preds[self.pred_idx][i].asnumpy()
            # calculate for each class
            while (pred.shape[0] > 0):
                cid = int(pred[0, 0])
                indices = np.where(pred[:, 0].astype(int) == cid)[0]
                if cid < 0:
                    pred = np.delete(pred, indices, axis=0)
                    continue
                dets = pred[indices]
                pred = np.delete(pred, indices, axis=0)
                # sort by score, desceding
                dets[dets[:,1].argsort()[::-1]]
                records = np.hstack((dets[:, 1][:, np.newaxis], np.zeros((dets.shape[0], 1))))
                # ground-truths
                label_indices = np.where(label[:, 0].astype(int) == cid)[0]
                gts = label[label_indices, :]
                label = np.delete(label, label_indices, axis=0)
                if gts.size > 0:
                    found = [False] * gts.shape[0]
                    for j in range(dets.shape[0]):
                        # compute overlaps
                        ious = iou(dets[j, 2:], gts[:, 1:5])
                        ovargmax = np.argmax(ious)
                        ovmax = ious[ovargmax]
                        if ovmax > self.ovp_thresh:
                            if (not self.use_difficult and
                                gts.shape[1] >= 6 and
                                gts[ovargmax, 5] > 0):
                                pass
                            else:
                                if not found[ovargmax]:
                                    records[j, -1] = 1  # tp
                                    found[ovargmax] = True
                                else:
                                    # duplicate
                                    records[j, -1] = 2  # fp
                        else:
                            records[j, -1] = 2 # fp
                else:
                    # no gt, mark all fp
                    records[:, -1] = 2

                # ground truth count
                if (not self.use_difficult and gts.shape[1] >= 6):
                    gt_count = np.sum(gts[:, 5] < 1)
                else:
                    gt_count = gts.shape[0]

                # now we push records to buffer
                # first column: score, second column: tp/fp
                # 0: not set(matched to difficult or something), 1: tp, 2: fp
                records = records[np.where(records[:, -1] > 0)[0], :]
                if records.size > 0:
                    self._insert(cid, records, gt_count)

            # add missing class if not present in prediction
            while (label.shape[0] > 0):
                cid = int(label[0, 0])
                label_indices = np.where(label[:, 0].astype(int) == cid)[0]
                label = np.delete(label, label_indices, axis=0)
                if cid < 0:
                    continue
                gt_count = label_indices.size
                self._insert(cid, np.array([[0, 0]]), gt_count)

    def _update(self):
        """ update num_inst and sum_metric """
        aps = []
        for k, v in self.records.items():
            recall, prec = self._recall_prec(v, self.counts[k])
            ap = self._average_precision(recall, prec)
            aps.append(ap)
            if self.num is not None and k < (self.num - 1):
                self.sum_metric[k] = ap
                self.num_inst[k] = 1
        if self.num is None:
            self.num_inst = 1
            self.sum_metric = np.mean(aps)
        else:
            self.num_inst[-1] = 1
            self.sum_metric[-1] = np.mean(aps)

    def _recall_prec(self, record, count):
        """ get recall and precision from internal records """
        record = np.delete(record, np.where(record[:, 1].astype(int) == 0)[0], axis=0)
        sorted_records = record[record[:,0].argsort()[::-1]]
        tp = np.cumsum(sorted_records[:, 1].astype(int) == 1)
        fp = np.cumsum(sorted_records[:, 1].astype(int) == 2)
        if count <= 0:
            recall = tp * 0.0
        else:
            recall = tp / float(count)
        prec = tp.astype(float) / (tp + fp)
        return recall, prec

    def _average_precision(self, rec, prec):
        """
        calculate average precision

        Params:
        ----------
        rec : numpy.array
            cumulated recall
        prec : numpy.array
            cumulated precision
        Returns:
        ----------
        ap as float
        """
        # append sentinel values at both ends
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute precision integration ladder
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # look for recall value changes
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # sum (\delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    def _insert(self, key, records, count):
        """ Insert records according to key """
        if key not in self.records:
            assert key not in self.counts
            self.records[key] = records
            self.counts[key] = count
        else:
            self.records[key] = np.vstack((self.records[key], records))
            assert key in self.counts
            self.counts[key] += count

class VOC07MApMetric(MApMetric):
    """ Mean average precision metric for PASCAL V0C 07 dataset """
    def __init__(self, *args, **kwargs):
        super(VOC07MApMetric, self).__init__(*args, **kwargs)

    def _average_precision(self, rec, prec):
        """
        calculate average precision, override the default one,
        special 11-point metric

        Params:
        ----------
        rec : numpy.array
            cumulated recall
        prec : numpy.array
            cumulated precision
        Returns:
        ----------
        ap as float
        """
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11.
        return ap


# In[10]:


import collections

class DotDict(dict):
    """
    Simple class for dot access elements in dict, support nested initialization
    Example:
    d = DotDict({'child': 'dotdict'}, name='dotdict', index=1, contents=['a', 'b'])
    # add new key
    d.new_key = '!' # or d['new_key'] = '!'
    # update values
    d.new_key = '!!!'
    # delete keys
    del d.new_key
    """
    def __init__(self, *args, **kwargs):
        super(DotDict, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        super(DotDict, self).__setitem__(key, value)
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        super(DotDict, self).__delitem__(key)
        del self.__dict__[key]


def namedtuple_with_defaults(typename, field_names, default_values=()):
    """ create a namedtuple with default values """
    T = collections.namedtuple(typename, field_names)
    T.__new__.__defaults__ = (None, ) * len(T._fields)
    if isinstance(default_values, collections.Mapping):
        prototype = T(**default_values)
    else:
        prototype = T(*default_values)
    T.__new__.__defaults__ = tuple(prototype)
    return T

def merge_dict(a, b):
    """ merge dict a, b, with b overriding keys in a """
    c = a.copy()
    c.update(b)
    return c

def zip_namedtuple(nt_list):
    """ accept list of namedtuple, return a dict of zipped fields """
    if not nt_list:
        return dict()
    if not isinstance(nt_list, list):
        nt_list = [nt_list]
    for nt in nt_list:
        assert type(nt) == type(nt_list[0])
    ret = {k : [v] for k, v in nt_list[0]._asdict().items()}
    for nt in nt_list[1:]:
        for k, v in nt._asdict().items():
            ret[k].append(v)
    return ret

def config_as_dict(cfg):
    """ convert raw configuration to unified dictionary """
    ret = cfg.__dict__.copy()
    # random cropping params
    del ret['rand_crop_samplers']
    assert isinstance(cfg.rand_crop_samplers, list)
    ret = merge_dict(ret, zip_namedtuple(cfg.rand_crop_samplers))
    num_crop_sampler = len(cfg.rand_crop_samplers)
    ret['num_crop_sampler'] = num_crop_sampler  # must specify the #
    ret['rand_crop_prob'] = 1.0 / (num_crop_sampler + 1) * num_crop_sampler
    # random padding params
    del ret['rand_pad']
    ret = merge_dict(ret, cfg.rand_pad._asdict())
    # color jitter
    del ret['color_jitter']
    ret = merge_dict(ret, cfg.color_jitter._asdict())
    return ret


# In[11]:


def get_symbol(num_classes=1000, **kwargs):
    """
    VGG 16 layers network
    This is a modified version, with fc6/fc7 layers replaced by conv layers
    And the network is slightly smaller than original VGG 16 network
    """
    data = mx.symbol.Variable(name="data")
    label = mx.symbol.Variable(name="label")

    # group 1
    conv1_1 = mx.symbol.Convolution(
        data=data, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_1")
    relu1_1 = mx.symbol.Activation(data=conv1_1, act_type="relu", name="relu1_1")
    conv1_2 = mx.symbol.Convolution(
        data=relu1_1, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_2")
    relu1_2 = mx.symbol.Activation(data=conv1_2, act_type="relu", name="relu1_2")
    pool1 = mx.symbol.Pooling(
        data=relu1_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool1")
    # group 2
    conv2_1 = mx.symbol.Convolution(
        data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_1")
    relu2_1 = mx.symbol.Activation(data=conv2_1, act_type="relu", name="relu2_1")
    conv2_2 = mx.symbol.Convolution(
        data=relu2_1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_2")
    relu2_2 = mx.symbol.Activation(data=conv2_2, act_type="relu", name="relu2_2")
    pool2 = mx.symbol.Pooling(
        data=relu2_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool2")
    # group 3
    conv3_1 = mx.symbol.Convolution(
        data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_1")
    relu3_1 = mx.symbol.Activation(data=conv3_1, act_type="relu", name="relu3_1")
    conv3_2 = mx.symbol.Convolution(
        data=relu3_1, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_2")
    relu3_2 = mx.symbol.Activation(data=conv3_2, act_type="relu", name="relu3_2")
    conv3_3 = mx.symbol.Convolution(
        data=relu3_2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_3")
    relu3_3 = mx.symbol.Activation(data=conv3_3, act_type="relu", name="relu3_3")
    pool3 = mx.symbol.Pooling(
        data=relu3_3, pool_type="max", kernel=(2, 2), stride=(2, 2), \
        pooling_convention="full", name="pool3")
    # group 4
    conv4_1 = mx.symbol.Convolution(
        data=pool3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_1")
    relu4_1 = mx.symbol.Activation(data=conv4_1, act_type="relu", name="relu4_1")
    conv4_2 = mx.symbol.Convolution(
        data=relu4_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_2")
    relu4_2 = mx.symbol.Activation(data=conv4_2, act_type="relu", name="relu4_2")
    conv4_3 = mx.symbol.Convolution(
        data=relu4_2, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_3")
    relu4_3 = mx.symbol.Activation(data=conv4_3, act_type="relu", name="relu4_3")
    pool4 = mx.symbol.Pooling(
        data=relu4_3, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool4")
    # group 5
    conv5_1 = mx.symbol.Convolution(
        data=pool4, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_1")
    relu5_1 = mx.symbol.Activation(data=conv5_1, act_type="relu", name="relu5_1")
    conv5_2 = mx.symbol.Convolution(
        data=relu5_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_2")
    relu5_2 = mx.symbol.Activation(data=conv5_2, act_type="relu", name="relu5_2")
    conv5_3 = mx.symbol.Convolution(
        data=relu5_2, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_3")
    relu5_3 = mx.symbol.Activation(data=conv5_3, act_type="relu", name="relu5_3")
    pool5 = mx.symbol.Pooling(
        data=relu5_3, pool_type="max", kernel=(3, 3), stride=(1, 1),
        pad=(1,1), name="pool5")
    # group 6
    conv6 = mx.symbol.Convolution(
        data=pool5, kernel=(3, 3), pad=(6, 6), dilate=(6, 6),
        num_filter=1024, name="fc6")
    relu6 = mx.symbol.Activation(data=conv6, act_type="relu", name="relu6")
    # drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    # group 7
    conv7 = mx.symbol.Convolution(
        data=relu6, kernel=(1, 1), pad=(0, 0), num_filter=1024, name="fc7")
    relu7 = mx.symbol.Activation(data=conv7, act_type="relu", name="relu7")
    # drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7")

    gpool = mx.symbol.Pooling(data=relu7, pool_type='avg', kernel=(7, 7),
        global_pool=True, name='global_pool')
    conv8 = mx.symbol.Convolution(data=gpool, num_filter=num_classes, kernel=(1, 1),
        name='fc8')
    flat = mx.symbol.Flatten(data=conv8)
    softmax = mx.symbol.SoftmaxOutput(data=flat, name='softmax')
    return softmax


# In[12]:


def conv_act_layer(from_layer, name, num_filter, kernel=(1,1), pad=(0,0),     stride=(1,1), act_type="relu", use_batchnorm=False):
    """
    wrapper for a small Convolution group

    Parameters:
    ----------
    from_layer : mx.symbol
        continue on which layer
    name : str
        base name of the new layers
    num_filter : int
        how many filters to use in Convolution layer
    kernel : tuple (int, int)
        kernel size (h, w)
    pad : tuple (int, int)
        padding size (h, w)
    stride : tuple (int, int)
        stride size (h, w)
    act_type : str
        activation type, can be relu...
    use_batchnorm : bool
        whether to use batch normalization

    Returns:
    ----------
    (conv, relu) mx.Symbols
    """
    conv = mx.symbol.Convolution(data=from_layer, kernel=kernel, pad=pad,         stride=stride, num_filter=num_filter, name="{}_conv".format(name))
    if use_batchnorm:
        conv = mx.symbol.BatchNorm(data=conv, name="{}_bn".format(name))
    relu = mx.symbol.Activation(data=conv, act_type=act_type,         name="{}_{}".format(name, act_type))
    return relu

def multi_layer_feature(body, from_layers, num_filters, strides, pads, min_filter=128):
    """Wrapper function to extract features from base network, attaching extra
    layers and SSD specific layers

    Parameters
    ----------
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    min_filter : int
        minimum number of filters used in 1x1 convolution

    Returns
    -------
    list of mx.Symbols

    """
    # arguments check
    assert len(from_layers) > 0
    assert isinstance(from_layers[0], str) and len(from_layers[0].strip()) > 0
    assert len(from_layers) == len(num_filters) == len(strides) == len(pads)

    internals = body.get_internals()
    layers = []
    for k, params in enumerate(zip(from_layers, num_filters, strides, pads)):
        from_layer, num_filter, s, p = params
        if from_layer.strip():
            # extract from base network
            layer = internals[from_layer.strip() + '_output']
            layers.append(layer)
        else:
            # attach from last feature layer
            assert len(layers) > 0
            assert num_filter > 0
            layer = layers[-1]
            num_1x1 = max(min_filter, num_filter // 2)
            conv_1x1 = conv_act_layer(layer, 'multi_feat_%d_conv_1x1' % (k),
                num_1x1, kernel=(1, 1), pad=(0, 0), stride=(1, 1), act_type='relu')
            conv_3x3 = conv_act_layer(conv_1x1, 'multi_feat_%d_conv_3x3' % (k),
                num_filter, kernel=(3, 3), pad=(p, p), stride=(s, s), act_type='relu')
            layers.append(conv_3x3)
    return layers

def multibox_layer(from_layers, num_classes, sizes=[.2, .95],
                    ratios=[1], normalization=-1, num_channels=[],
                    clip=False, interm_layer=0, steps=[]):
    """
    the basic aggregation module for SSD detection. Takes in multiple layers,
    generate multiple object detection targets by customized layers

    Parameters:
    ----------
    from_layers : list of mx.symbol
        generate multibox detection from layers
    num_classes : int
        number of classes excluding background, will automatically handle
        background in this function
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    num_channels : list of int
        number of input layer channels, used when normalization is enabled, the
        length of list should equals to number of normalization layers
    clip : bool
        whether to clip out-of-image boxes
    interm_layer : int
        if > 0, will add a intermediate Convolution layer
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions

    Returns:
    ----------
    list of outputs, as [loc_preds, cls_preds, anchor_boxes]
    loc_preds : localization regression prediction
    cls_preds : classification prediction
    anchor_boxes : generated anchor boxes
    """
    assert len(from_layers) > 0, "from_layers must not be empty list"
    assert num_classes > 0,         "num_classes {} must be larger than 0".format(num_classes)

    assert len(ratios) > 0, "aspect ratios must not be empty list"
    if not isinstance(ratios[0], list):
        # provided only one ratio list, broadcast to all from_layers
        ratios = [ratios] * len(from_layers)
    assert len(ratios) == len(from_layers),         "ratios and from_layers must have same length"

    assert len(sizes) > 0, "sizes must not be empty list"
    if len(sizes) == 2 and not isinstance(sizes[0], list):
        # provided size range, we need to compute the sizes for each layer
         assert sizes[0] > 0 and sizes[0] < 1
         assert sizes[1] > 0 and sizes[1] < 1 and sizes[1] > sizes[0]
         tmp = np.linspace(sizes[0], sizes[1], num=(len(from_layers)-1))
         min_sizes = [start_offset] + tmp.tolist()
         max_sizes = tmp.tolist() + [tmp[-1]+start_offset]
         sizes = zip(min_sizes, max_sizes)
    assert len(sizes) == len(from_layers),         "sizes and from_layers must have same length"

    if not isinstance(normalization, list):
        normalization = [normalization] * len(from_layers)
    assert len(normalization) == len(from_layers)

    assert sum(x > 0 for x in normalization) <= len(num_channels),         "must provide number of channels for each normalized layer"

    if steps:
        assert len(steps) == len(from_layers), "provide steps for all layers or leave empty"

    loc_pred_layers = []
    cls_pred_layers = []
    anchor_layers = []
    num_classes += 1 # always use background as label 0

    for k, from_layer in enumerate(from_layers):
        from_name = from_layer.name
        # normalize
        if normalization[k] > 0:
            from_layer = mx.symbol.L2Normalization(data=from_layer,                 mode="channel", name="{}_norm".format(from_name))
            scale = mx.symbol.Variable(name="{}_scale".format(from_name),
                shape=(1, num_channels.pop(0), 1, 1),
                init=mx.init.Constant(normalization[k]),
                attr={'__wd_mult__': '0.1'})
            from_layer = mx.symbol.broadcast_mul(lhs=scale, rhs=from_layer)
        if interm_layer > 0:
            from_layer = mx.symbol.Convolution(data=from_layer, kernel=(3,3),                 stride=(1,1), pad=(1,1), num_filter=interm_layer,                 name="{}_inter_conv".format(from_name))
            from_layer = mx.symbol.Activation(data=from_layer, act_type="relu",                 name="{}_inter_relu".format(from_name))

        # estimate number of anchors per location
        # here I follow the original version in caffe
        # TODO: better way to shape the anchors??
        size = sizes[k]
        assert len(size) > 0, "must provide at least one size"
        size_str = "(" + ",".join([str(x) for x in size]) + ")"
        ratio = ratios[k]
        assert len(ratio) > 0, "must provide at least one ratio"
        ratio_str = "(" + ",".join([str(x) for x in ratio]) + ")"
        num_anchors = len(size) -1 + len(ratio)

        # create location prediction layer
        num_loc_pred = num_anchors * 4
        bias = mx.symbol.Variable(name="{}_loc_pred_conv_bias".format(from_name),
            init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0'})
        loc_pred = mx.symbol.Convolution(data=from_layer, bias=bias, kernel=(3,3),             stride=(1,1), pad=(1,1), num_filter=num_loc_pred,             name="{}_loc_pred_conv".format(from_name))
        loc_pred = mx.symbol.transpose(loc_pred, axes=(0,2,3,1))
        loc_pred = mx.symbol.Flatten(data=loc_pred)
        loc_pred_layers.append(loc_pred)

        # create class prediction layer
        num_cls_pred = num_anchors * num_classes
        bias = mx.symbol.Variable(name="{}_cls_pred_conv_bias".format(from_name),
            init=mx.init.Constant(0.0), attr={'__lr_mult__': '2.0'})
        cls_pred = mx.symbol.Convolution(data=from_layer, bias=bias, kernel=(3,3),             stride=(1,1), pad=(1,1), num_filter=num_cls_pred,             name="{}_cls_pred_conv".format(from_name))
        cls_pred = mx.symbol.transpose(cls_pred, axes=(0,2,3,1))
        cls_pred = mx.symbol.Flatten(data=cls_pred)
        cls_pred_layers.append(cls_pred)

        # create anchor generation layer
        if steps:
            step = (steps[k], steps[k])
        else:
            step = '(-1.0, -1.0)'
        anchors = mx.symbol.contrib.MultiBoxPrior(from_layer, sizes=size_str, ratios=ratio_str,
                                                  clip=clip, name="{}_anchors".format(from_name),
                                                  steps=step)
        anchors = mx.symbol.Flatten(data=anchors)
        anchor_layers.append(anchors)

    loc_preds = mx.symbol.Concat(*loc_pred_layers, num_args=len(loc_pred_layers),         dim=1, name="multibox_loc_pred")
    cls_preds = mx.symbol.Concat(*cls_pred_layers, num_args=len(cls_pred_layers),         dim=1)
    cls_preds = mx.symbol.Reshape(data=cls_preds, shape=(0, -1, num_classes))
    cls_preds = mx.symbol.transpose(cls_preds, axes=(0, 2, 1), name="multibox_cls_pred")
    anchor_boxes = mx.symbol.Concat(*anchor_layers,         num_args=len(anchor_layers), dim=1)
    anchor_boxes = mx.symbol.Reshape(data=anchor_boxes, shape=(0, -1, 4), name="multibox_anchors")
    return [loc_preds, cls_preds, anchor_boxes]


# In[13]:


def get_symbol_train(network, num_classes, from_layers, num_filters, strides, pads,
                     sizes, ratios, normalizations=-1, steps=[], min_filter=128,
                     nms_thresh=0.5, force_suppress=False, nms_topk=400, **kwargs):
    """Build network symbol for training SSD

    Parameters
    ----------
    network : str
        base network symbol name
    num_classes : int
        number of object classes not including background
    from_layers : list of str
        feature extraction layers, use '' for add extra layers
        For example:
        from_layers = ['relu4_3', 'fc7', '', '', '', '']
        which means extract feature from relu4_3 and fc7, adding 4 extra layers
        on top of fc7
    num_filters : list of int
        number of filters for extra layers, you can use -1 for extracted features,
        however, if normalization and scale is applied, the number of filter for
        that layer must be provided.
        For example:
        num_filters = [512, -1, 512, 256, 256, 256]
    strides : list of int
        strides for the 3x3 convolution appended, -1 can be used for extracted
        feature layers
    pads : list of int
        paddings for the 3x3 convolution, -1 can be used for extracted layers
    sizes : list or list of list
        [min_size, max_size] for all layers or [[], [], []...] for specific layers
    ratios : list or list of list
        [ratio1, ratio2...] for all layers or [[], [], ...] for specific layers
    normalizations : int or list of int
        use normalizations value for all layers or [...] for specific layers,
        -1 indicate no normalizations and scales
    steps : list
        specify steps for each MultiBoxPrior layer, leave empty, it will calculate
        according to layer dimensions
    min_filter : int
        minimum number of filters used in 1x1 convolution
    nms_thresh : float
        non-maximum suppression threshold
    force_suppress : boolean
        whether suppress different class objects
    nms_topk : int
        apply NMS to top K detections

    Returns
    -------
    mx.Symbol

    """
    label = mx.sym.Variable('label')
    body = get_symbol(num_classes, **kwargs)
    layers = multi_layer_feature(body, from_layers, num_filters, strides, pads,
        min_filter=min_filter)

    loc_preds, cls_preds, anchor_boxes = multibox_layer(layers,         num_classes, sizes=sizes, ratios=ratios, normalization=normalizations,         num_channels=num_filters, clip=False, interm_layer=0, steps=steps)

    tmp = mx.symbol.contrib.MultiBoxTarget(
        *[anchor_boxes, label, cls_preds], overlap_threshold=.5, \
        ignore_label=-1, negative_mining_ratio=3, minimum_negative_samples=0, \
        negative_mining_thresh=.5, variances=(0.1, 0.1, 0.2, 0.2),
        name="multibox_target")
    loc_target = tmp[0]
    loc_target_mask = tmp[1]
    cls_target = tmp[2]

    cls_prob = mx.symbol.SoftmaxOutput(data=cls_preds, label=cls_target,         ignore_label=-1, use_ignore=True, grad_scale=1., multi_output=True,         normalization='valid', name="cls_prob")
    loc_loss_ = mx.symbol.smooth_l1(name="loc_loss_",         data=loc_target_mask * (loc_preds - loc_target), scalar=1.0)
    loc_loss = mx.symbol.MakeLoss(loc_loss_, grad_scale=1.,         normalization='valid', name="loc_loss")

    # monitoring training status
    cls_label = mx.symbol.MakeLoss(data=cls_target, grad_scale=0, name="cls_label")
    det = mx.symbol.contrib.MultiBoxDetection(*[cls_prob, loc_preds, anchor_boxes],         name="detection", nms_threshold=nms_thresh, force_suppress=force_suppress,
        variances=(0.1, 0.1, 0.2, 0.2), nms_topk=nms_topk)
    det = mx.symbol.MakeLoss(data=det, grad_scale=0, name="det_out")

    # group output
    out = mx.symbol.Group([cls_prob, loc_loss, cls_label, det])
    return out


# In[14]:


def get_config(network, data_shape, **kwargs):
    """Configuration factory for various networks

    Parameters
    ----------
    network : str
        base network name, such as vgg_reduced, inceptionv3, resnet...
    data_shape : int
        input data dimension
    kwargs : dict
        extra arguments
    """
    if network == 'vgg16_reduced':
        if data_shape >= 448:
            from_layers = ['relu4_3', 'relu7', '', '', '', '', '']
            num_filters = [512, -1, 512, 256, 256, 256, 256]
            strides = [-1, -1, 2, 2, 2, 2, 1]
            pads = [-1, -1, 1, 1, 1, 1, 1]
            sizes = [[.07, .1025], [.15,.2121], [.3, .3674], [.45, .5196], [.6, .6708],                 [.75, .8216], [.9, .9721]]
            ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3],                 [1,2,.5,3,1./3], [1,2,.5], [1,2,.5]]
            normalizations = [20, -1, -1, -1, -1, -1, -1]
            steps = [] if data_shape != 512 else [x / 512.0 for x in
                [8, 16, 32, 64, 128, 256, 512]]
        else:
            from_layers = ['relu4_3', 'relu7', '', '', '', '']
            num_filters = [512, -1, 512, 256, 256, 256]
            strides = [-1, -1, 2, 2, 1, 1]
            pads = [-1, -1, 1, 1, 0, 0]
            sizes = [[.1, .141], [.2,.272], [.37, .447], [.54, .619], [.71, .79], [.88, .961]]
            ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3],                 [1,2,.5], [1,2,.5]]
            normalizations = [20, -1, -1, -1, -1, -1]
            steps = [] if data_shape != 300 else [x / 300.0 for x in [8, 16, 32, 64, 100, 300]]
        if not (data_shape == 300 or data_shape == 512):
            logging.warn('data_shape %d was not tested, use with caucious.' % data_shape)
        return locals()
    elif network == 'inceptionv3':
        from_layers = ['ch_concat_mixed_7_chconcat', 'ch_concat_mixed_10_chconcat', '', '', '', '']
        num_filters = [-1, -1, 512, 256, 256, 128]
        strides = [-1, -1, 2, 2, 2, 2]
        pads = [-1, -1, 1, 1, 1, 1]
        sizes = [[.1, .141], [.2,.272], [.37, .447], [.54, .619], [.71, .79], [.88, .961]]
        ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3],             [1,2,.5], [1,2,.5]]
        normalizations = -1
        steps = []
        return locals()
    elif network == 'resnet50':
        num_layers = 50
        image_shape = '3,224,224'  # resnet require it as shape check
        network = 'resnet'
        from_layers = ['_plus12', '_plus15', '', '', '', '']
        num_filters = [-1, -1, 512, 256, 256, 128]
        strides = [-1, -1, 2, 2, 2, 2]
        pads = [-1, -1, 1, 1, 1, 1]
        sizes = [[.1, .141], [.2,.272], [.37, .447], [.54, .619], [.71, .79], [.88, .961]]
        ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3],             [1,2,.5], [1,2,.5]]
        normalizations = -1
        steps = []
        return locals()
    elif network == 'resnet101':
        num_layers = 101
        image_shape = '3,224,224'
        network = 'resnet'
        from_layers = ['_plus29', '_plus32', '', '', '', '']
        num_filters = [-1, -1, 512, 256, 256, 128]
        strides = [-1, -1, 2, 2, 2, 2]
        pads = [-1, -1, 1, 1, 1, 1]
        sizes = [[.1, .141], [.2,.272], [.37, .447], [.54, .619], [.71, .79], [.88, .961]]
        ratios = [[1,2,.5], [1,2,.5,3,1./3], [1,2,.5,3,1./3], [1,2,.5,3,1./3],             [1,2,.5], [1,2,.5]]
        normalizations = -1
        steps = []
        return locals()
    else:
        msg = 'No configuration found for %s with data_shape %d' % (network, data_shape)
        raise NotImplementedError(msg)

def get_network_train(network, data_shape, **kwargs):
    """Wrapper for get symbol for train

    Parameters
    ----------
    network : str
        name for the base network symbol
    data_shape : int
        input shape
    kwargs : dict
        see symbol_builder.get_symbol_train for more details
    """
    config = get_config(network, data_shape, **kwargs).copy()
    config.update(kwargs)
    return get_symbol_train(**config)


# In[15]:


def convert_pretrained(name, args):
    """
    Special operations need to be made due to name inconsistance, etc

    Parameters:
    ---------
    name : str
        pretrained model name
    args : dict
        loaded arguments

    Returns:
    ---------
    processed arguments as dict
    """
    return args

def get_lr_scheduler(learning_rate, lr_refactor_step, lr_refactor_ratio,
                     num_example, batch_size, begin_epoch):
    """
    Compute learning rate and refactor scheduler

    Parameters:
    ---------
    learning_rate : float
        original learning rate
    lr_refactor_step : comma separated str
        epochs to change learning rate
    lr_refactor_ratio : float
        lr *= ratio at certain steps
    num_example : int
        number of training images, used to estimate the iterations given epochs
    batch_size : int
        training batch size
    begin_epoch : int
        starting epoch

    Returns:
    ---------
    (learning_rate, mx.lr_scheduler) as tuple
    """
    assert lr_refactor_ratio > 0
    iter_refactor = [int(r) for r in lr_refactor_step.split(',') if r.strip()]
    if lr_refactor_ratio >= 1:
        return (learning_rate, None)
    else:
        lr = learning_rate
        epoch_size = num_example // batch_size
        for s in iter_refactor:
            if begin_epoch >= s:
                lr *= lr_refactor_ratio
        if lr != learning_rate:
            logging.getLogger().info("Adjusted learning rate to {} for epoch {}".format(lr, begin_epoch))
        steps = [epoch_size * (x - begin_epoch) for x in iter_refactor if x > begin_epoch]
        if not steps:
            return (lr, None)
        lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=lr_refactor_ratio)
        return (lr, lr_scheduler)

def train_net(net, train_path, num_classes, batch_size,
              data_shape, mean_pixels, resume, finetune, pretrained, epoch,
              prefix, ctx, begin_epoch, end_epoch, frequent, learning_rate,
              momentum, weight_decay, lr_refactor_step, lr_refactor_ratio,
              freeze_layer_pattern='',
              num_example=10000, label_pad_width=350,
              nms_thresh=0.45, force_nms=False, ovp_thresh=0.5,
              use_difficult=False, class_names=None,
              voc07_metric=False, nms_topk=400, force_suppress=False,
              train_list="", val_path="", val_list="", iter_monitor=0,
              monitor_pattern=".*", log_file=None):
    """
    Wrapper for training phase.

    Parameters:
    ----------
    net : str
        symbol name for the network structure
    train_path : str
        record file path for training
    num_classes : int
        number of object classes, not including background
    batch_size : int
        training batch-size
    data_shape : int or tuple
        width/height as integer or (3, height, width) tuple
    mean_pixels : tuple of floats
        mean pixel values for red, green and blue
    resume : int
        resume from previous checkpoint if > 0
    finetune : int
        fine-tune from previous checkpoint if > 0
    pretrained : str
        prefix of pretrained model, including path
    epoch : int
        load epoch of either resume/finetune/pretrained model
    prefix : str
        prefix for saving checkpoints
    ctx : [mx.cpu()] or [mx.gpu(x)]
        list of mxnet contexts
    begin_epoch : int
        starting epoch for training, should be 0 if not otherwise specified
    end_epoch : int
        end epoch of training
    frequent : int
        frequency to print out training status
    learning_rate : float
        training learning rate
    momentum : float
        trainig momentum
    weight_decay : float
        training weight decay param
    lr_refactor_ratio : float
        multiplier for reducing learning rate
    lr_refactor_step : comma separated integers
        at which epoch to rescale learning rate, e.g. '30, 60, 90'
    freeze_layer_pattern : str
        regex pattern for layers need to be fixed
    num_example : int
        number of training images
    label_pad_width : int
        force padding training and validation labels to sync their label widths
    nms_thresh : float
        non-maximum suppression threshold for validation
    force_nms : boolean
        suppress overlaped objects from different classes
    train_list : str
        list file path for training, this will replace the embeded labels in record
    val_path : str
        record file path for validation
    val_list : str
        list file path for validation, this will replace the embeded labels in record
    iter_monitor : int
        monitor internal stats in networks if > 0, specified by monitor_pattern
    monitor_pattern : str
        regex pattern for monitoring network stats
    log_file : str
        log to file if enabled
    """
    # set up logger
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if log_file:
        fh = logging.FileHandler(log_file)
        logger.addHandler(fh)

    # check args
    if isinstance(data_shape, int):
        data_shape = (3, data_shape, data_shape)
    assert len(data_shape) == 3 and data_shape[0] == 3
    prefix += '_' + net + '_' + str(data_shape[1])

    if isinstance(mean_pixels, (int, float)):
        mean_pixels = [mean_pixels, mean_pixels, mean_pixels]
    assert len(mean_pixels) == 3, "must provide all RGB mean values"

    RandCropper = namedtuple_with_defaults('RandCropper',
        'min_crop_scales, max_crop_scales, \
        min_crop_aspect_ratios, max_crop_aspect_ratios, \
        min_crop_overlaps, max_crop_overlaps, \
        min_crop_sample_coverages, max_crop_sample_coverages, \
        min_crop_object_coverages, max_crop_object_coverages, \
        max_crop_trials',
        [0.0, 1.0,
        0.5, 2.0,
        0.0, 1.0,
        0.0, 1.0,
        0.0, 1.0,
        25])

    RandPadder = namedtuple_with_defaults('RandPadder',
        'rand_pad_prob, max_pad_scale, fill_value',
        [0.0, 1.0, 127])

    ColorJitter = namedtuple_with_defaults('ColorJitter',
        'random_hue_prob, max_random_hue, \
        random_saturation_prob, max_random_saturation, \
        random_illumination_prob, max_random_illumination, \
        random_contrast_prob, max_random_contrast',
        [0.0, 18,
        0.0, 32,
        0.0, 32,
        0.0, 0.5])

    cfg = DotDict()
    cfg.ROOT_DIR = os.getcwd()

    # training configs
    cfg.train = DotDict()
    # random cropping samplers
    cfg.train.rand_crop_samplers = [
        RandCropper(min_crop_scales=0.3, min_crop_overlaps=0.1),
        RandCropper(min_crop_scales=0.3, min_crop_overlaps=0.3),
        RandCropper(min_crop_scales=0.3, min_crop_overlaps=0.5),
        RandCropper(min_crop_scales=0.3, min_crop_overlaps=0.7),
        RandCropper(min_crop_scales=0.3, min_crop_overlaps=0.9),]
    cfg.train.crop_emit_mode = 'center'
    # cfg.train.emit_overlap_thresh = 0.4
    # random padding
    cfg.train.rand_pad = RandPadder(rand_pad_prob=0.5, max_pad_scale=4.0)
    # random color jitter
    cfg.train.color_jitter = ColorJitter(random_hue_prob=0.5, random_saturation_prob=0.5,
        random_illumination_prob=0.5, random_contrast_prob=0.5)
    cfg.train.inter_method = 10  # random interpolation
    cfg.train.rand_mirror_prob = 0.5
    cfg.train.shuffle = True
    cfg.train.seed = 233
    cfg.train.preprocess_threads = 48
    cfg.train = config_as_dict(cfg.train)  # convert to normal dict

    # validation
    cfg.valid = DotDict()
    cfg.valid.rand_crop_samplers = []
    cfg.valid.rand_pad = RandPadder()
    cfg.valid.color_jitter = ColorJitter()
    cfg.valid.rand_mirror_prob = 0
    cfg.valid.shuffle = False
    cfg.valid.seed = 0
    cfg.valid.preprocess_threads = 32
    cfg.valid = config_as_dict(cfg.valid)  # convert to normal dict

    train_iter = DetRecordIter(train_path, batch_size, data_shape, mean_pixels=mean_pixels,
        label_pad_width=label_pad_width, path_imglist=train_list, **cfg.train)

    if val_path:
        val_iter = DetRecordIter(val_path, batch_size, data_shape, mean_pixels=mean_pixels,
            label_pad_width=label_pad_width, path_imglist=val_list, **cfg.valid)
    else:
        val_iter = None

    # load symbol
    net = get_network_train(net, data_shape[1], num_classes=num_classes,
        nms_thresh=nms_thresh, force_suppress=force_suppress, nms_topk=nms_topk)

    # define layers with fixed weight/bias
    if freeze_layer_pattern.strip():
        re_prog = re.compile(freeze_layer_pattern)
        fixed_param_names = [name for name in net.list_arguments() if re_prog.match(name)]
    else:
        fixed_param_names = None

    # load pretrained or resume from previous state
    ctx_str = '('+ ','.join([str(c) for c in ctx]) + ')'
    if resume > 0:
        logger.info("Resume training with {} from epoch {}"
            .format(ctx_str, resume))
        _, args, auxs = mx.model.load_checkpoint(prefix, resume)
        begin_epoch = resume
    elif finetune > 0:
        logger.info("Start finetuning with {} from epoch {}"
            .format(ctx_str, finetune))
        _, args, auxs = mx.model.load_checkpoint(prefix, finetune)
        begin_epoch = finetune
        # the prediction convolution layers name starts with relu, so it's fine
        fixed_param_names = [name for name in net.list_arguments()             if name.startswith('conv')]
    elif pretrained:
        logger.info("Start training with {} from pretrained model {}"
            .format(ctx_str, pretrained))
        _, args, auxs = mx.model.load_checkpoint(pretrained, epoch)
        args = convert_pretrained(pretrained, args)
    else:
        logger.info("Experimental: start training from scratch with {}"
            .format(ctx_str))
        args = None
        auxs = None
        fixed_param_names = None

    # helper information
    if fixed_param_names:
        logger.info("Freezed parameters: [" + ','.join(fixed_param_names) + ']')

    # init training module
    mod = mx.mod.Module(net, label_names=('label',), logger=logger, context=ctx,
                        fixed_param_names=fixed_param_names)

    # fit parameters
    batch_end_callback = mx.callback.Speedometer(train_iter.batch_size, frequent=frequent)
    epoch_end_callback = mx.callback.do_checkpoint(prefix)
    learning_rate, lr_scheduler = get_lr_scheduler(learning_rate, lr_refactor_step,
        lr_refactor_ratio, num_example, batch_size, begin_epoch)
    optimizer_params={'learning_rate':learning_rate,
                      'momentum':momentum,
                      'wd':weight_decay,
                      'lr_scheduler':lr_scheduler,
                      'clip_gradient':None,
                      'rescale_grad': 1.0 / len(ctx) if len(ctx) > 0 else 1.0 }
    monitor = mx.mon.Monitor(iter_monitor, pattern=monitor_pattern) if iter_monitor > 0 else None

    # run fit net, every n epochs we run evaluation network to get mAP
    if voc07_metric:
        valid_metric = VOC07MApMetric(ovp_thresh, use_difficult, class_names, pred_idx=3)
    else:
        valid_metric = MApMetric(ovp_thresh, use_difficult, class_names, pred_idx=3)

    mod.symbol.save(os.path.join(os.getcwd(), 'model', 'ssd-symbol.json'))
        
    mod.fit(train_iter,
            val_iter,
            eval_metric=MultiBoxMetric(),
            validation_metric=valid_metric,
            batch_end_callback=batch_end_callback,
            epoch_end_callback=epoch_end_callback,
            optimizer='sgd',
            optimizer_params=optimizer_params,
            begin_epoch=begin_epoch,
            num_epoch=end_epoch,
            initializer=mx.init.Xavier(),
            arg_params=args,
            aux_params=auxs,
            allow_missing=True,
            monitor=monitor)


# In[ ]:


def parse_class_names(class_names, num_class):
    """ parse # classes and class_names if applicable """
    num_class = num_class
    if len(class_names) > 0:
        if os.path.isfile(class_names):
            # try to open it to read class names
            with open(class_names, 'r') as f:
                class_names = [l.strip() for l in f.readlines()]
        else:
            class_names = [c.strip() for c in class_names.split(',')]
        assert len(class_names) == num_class, str(len(class_names))
        for name in class_names:
            assert len(name) > 0
    else:
        class_names = None
    return class_names


# In[ ]:


# context list
ctx = [mx.cpu()]

class_names = parse_class_names('label/synset.txt', 30)

# start training
train_net(net='vgg16_reduced', train_path=os.path.join(os.getcwd(), 'data', 'train.rec'),
          num_classes=21, batch_size=32, data_shape=300, mean_pixels=[123, 117, 104], resume=-1,
          finetune=-1, pretrained=os.path.join(os.getcwd(), 'model', 'vgg16_reduced'),
          epoch=1, prefix=os.path.join(os.getcwd(), 'out', 'ssd'), ctx=ctx, begin_epoch=0, end_epoch=300,
          frequent=10, learning_rate=0.002, momentum=0.9, weight_decay=0.0005, lr_refactor_step='80, 160',
          lr_refactor_ratio=0.1, val_path=os.path.join(os.getcwd(), 'data', 'val.rec'),
          num_example=17791, class_names=class_names, label_pad_width=350, freeze_layer_pattern='^(conv1_|conv2_).*',
          iter_monitor=0, monitor_pattern='.*', log_file=os.path.join(os.getcwd(), 'log', 'train.log'),
          nms_thresh=0.45, force_nms=False, ovp_thresh=0.5, use_difficult=False, voc07_metric=True)


# In[18]:


get_ipython().run_cell_magic('bash', '', 'aws s3 sync out/ s3://ps-jiaan01-artifacts/out')

