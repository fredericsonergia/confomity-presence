from gluoncv.data import VOCDetection
from gluoncv import model_zoo
import mxnet as mx
from mxnet import autograd, gluon
import gluoncv as gcv
import numpy as np
from gluoncv.data.transforms import image as timage
from gluoncv.data.transforms import bbox as tbbox
from gluoncv.data.transforms import experimental

class VOCLike(VOCDetection):
    CLASSES = ["cheminee", "eaf"]

    def __init__(
        self, root, splits, transform=None, index_map=None, preload_label=True
    ):
        super(VOCLike, self).__init__(root, splits, transform, index_map, preload_label)


def new_trainloader_call(self, src, label):
    '''
    define a new call for trainloader by changing the data augmentation
    '''
    # random color jittering
    img = experimental.image.random_color_distort(src)

    # random expansion with prob 0.5
    if np.random.uniform(0, 1) > 0.5:
        img, expand = timage.random_expand(img, fill=[m * 255 for m in self._mean])
        bbox = tbbox.translate(label, x_offset=expand[0], y_offset=expand[1])
    else:
        img, bbox = img, label

    # random cropping
    h, w, _ = img.shape
    bbox, crop = experimental.bbox.random_crop_with_constraints(bbox, (w, h))
    x0, y0, w, h = crop
    img = mx.image.fixed_crop(img, x0, y0, w, h)

    # resize with random interpolation
    h, w, _ = img.shape
    interp = np.random.randint(0, 5)
    img = timage.imresize(img, self._width, self._height, interp=interp)
    bbox = tbbox.resize(bbox, (w, h), (self._width, self._height))

    # to tensor
    img = mx.nd.image.to_tensor(img)
    img = mx.nd.image.normalize(img, mean=self._mean, std=self._std)
    if self._anchors is None:
        return img, bbox.astype(img.dtype)

    # generate training target so cpu workers can help reduce the workload on gpu
    gt_bboxes = mx.nd.array(bbox[np.newaxis, :, :4])
    gt_ids = mx.nd.array(bbox[np.newaxis, :, 4:5])
    cls_targets, box_targets, _ = self._target_generator(
        self._anchors, None, gt_bboxes, gt_ids)
    return img, cls_targets[0], box_targets[0]

def ssd_train_dataloader(
    net, train_dataset, data_shape=512, batch_size=10, num_workers=0
):
    '''
    returns the train loader from gluoncv
    '''
    from gluoncv.data.batchify import Tuple, Stack, Pad
    from gluoncv.data.transforms.presets.ssd import SSDDefaultTrainTransform

    width, height = data_shape, data_shape
    # use fake data to generate fixed anchors for target generation
    with autograd.train_mode():
        _, _, anchors = net(mx.nd.zeros((1, 3, height, width)))
    batchify_fn = Tuple(
        Stack(), Stack(), Stack()
    )  # stack image, cls_targets, box_targets
    new_SSDDefaultTrainTransform=SSDDefaultTrainTransform(width, height, anchors)
    new_SSDDefaultTrainTransform.__call__= new_trainloader_call
    train_loader = gluon.data.DataLoader(
        train_dataset.transform(new_SSDDefaultTrainTransform),
        batch_size,
        True,
        batchify_fn=batchify_fn,
        last_batch="rollover",
        num_workers=num_workers,
    )
    return train_loader


def ssd_val_dataloader(val_dataset, data_shape=512, batch_size=10, num_workers=0):
    '''
    returns the validation loader from gluoncv
    '''
    from gluoncv.data.batchify import Tuple, Stack, Pad
    from gluoncv.data.transforms.presets.ssd import SSDDefaultValTransform

    width, height = data_shape, data_shape
    batchify_fn = Tuple(
        Stack(), Pad(pad_val=-1)
    )  # stack image, cls_targets, box_targets
    val_loader = gluon.data.DataLoader(
        val_dataset.transform(SSDDefaultValTransform(width, height)),
        batch_size,
        shuffle=False,
        batchify_fn=batchify_fn,
        last_batch="keep",
        num_workers=num_workers,
    )
    return val_loader


def validate(net, val_data, ctx, eval_metric, flip_test=False):
    """
    validation on MAP (mean average precision)
    """
    eval_metric.reset()
    net.flip_test = flip_test
    mx.nd.waitall()
    net.hybridize()
    for batch in val_data:
        data = gluon.utils.split_and_load(
            batch[0], ctx_list=ctx, batch_axis=0, even_split=False
        )
        label = gluon.utils.split_and_load(
            batch[1], ctx_list=ctx, batch_axis=0, even_split=False
        )
        det_bboxes = []
        det_ids = []
        det_scores = []
        gt_bboxes = []
        gt_ids = []
        gt_difficults = []
        for x, y in zip(data, label):
            # get prediction results
            ids, scores, bboxes = net(x)
            det_ids.append(ids)
            det_scores.append(scores)
            # clip to image size
            det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
            # split ground truths
            gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
            gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
            gt_difficults.append(
                y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None
            )

        # update metric
        eval_metric.update(
            det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults
        )
    return eval_metric.get()


def save_params(net, best_map, current_map, epoch, log_folder, prefix="ssd_512", model_folder='models/'):
    """
    save parameters of the networks
    Args:
    - net (network): the network to save
    - best_map (list): the current best map
    - current_map (float): the current map
    """
    current_map = float(current_map)
    if current_map > best_map[0]:
        best_map[0] = current_map
        net.save_parameters(model_folder + "{:s}_best.params".format(prefix))
        with open(log_folder + prefix + "_best_map.log", "a") as f:
            f.write("{:04d}:\t{:.4f}\n".format(epoch, current_map))


def get_ctx():
    '''
    get the context from mxnet
    '''
    try:
        a = mx.nd.zeros((1,), ctx=mx.gpu(0))
        ctx = [mx.gpu(0)]
    except:
        ctx = [mx.cpu()]
    return ctx


def val_loss(net, val_data, ctx):
    """
    validation on Loss.
    Args:
    - net (gluocv network): the network to validate
    - val_data (Dataloader, mini batch of data): the data on which we validate the model
    - ctx (array): gpu or cpu
    """
    mx.nd.waitall()
    net.hybridize()
    mbox_loss_val = gcv.loss.SSDMultiBoxLoss()
    ce_metric_val = mx.metric.Loss("CrossEntropy")
    smoothl1_metric_val = mx.metric.Loss("SmoothL1")
    ce_metric_val.reset()
    smoothl1_metric_val.reset()
    for batch in val_data:
        batch_size = batch[0].shape[0]
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        cls_targets = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        box_targets = gluon.utils.split_and_load(batch[2], ctx_list=ctx, batch_axis=0)
        with autograd.record():
            cls_preds = []
            box_preds = []
            for x in data:
                cls_pred, box_pred, _ = net(x)
                cls_preds.append(cls_pred)
                box_preds.append(box_pred)
            sum_loss, cls_loss, box_loss = mbox_loss_val(
                cls_preds, box_preds, cls_targets, box_targets
            )
        # since we have already normalized the loss, we don't want to normalize
        # by batch-size anymore
        ce_metric_val.update(0, [l * batch_size for l in cls_loss])
        smoothl1_metric_val.update(0, [l * batch_size for l in box_loss])
    name1, val_loss1 = ce_metric_val.get()
    name2, val_loss2 = smoothl1_metric_val.get()
    return val_loss1, val_loss2
