from gluoncv.data import VOCDetection
from gluoncv import model_zoo
import mxnet as mx
from mxnet import autograd, gluon
import gluoncv as gcv


class VOCLike(VOCDetection):
    CLASSES = ["cheminee", "eaf"]

    def __init__(
        self, root, splits, transform=None, index_map=None, preload_label=True
    ):
        super(VOCLike, self).__init__(root, splits, transform, index_map, preload_label)


def load_data_VOC(root="../Data/EAF_2labels"):
    dataset = VOCLike(root=root, splits=[(2021, "trainval")])
    val_dataset = VOCLike(root=root, splits=[(2021, "test")])
    return dataset, val_dataset


def get_pretrained_model(classes):
    # ssd_512_resnet50_v1_voc
    net = model_zoo.get_model(
        "ssd_512_mobilenet1.0_custom",
        classes=classes,
        pretrained_base=False,
        transfer="voc",
    )
    return net


def ssd_train_dataloader(
    net, train_dataset, data_shape=512, batch_size=10, num_workers=0
):
    from gluoncv.data.batchify import Tuple, Stack, Pad
    from gluoncv.data.transforms.presets.ssd import SSDDefaultTrainTransform

    width, height = data_shape, data_shape
    # use fake data to generate fixed anchors for target generation
    with autograd.train_mode():
        _, _, anchors = net(mx.nd.zeros((1, 3, height, width)))
    batchify_fn = Tuple(
        Stack(), Stack(), Stack()
    )  # stack image, cls_targets, box_targets
    train_loader = gluon.data.DataLoader(
        train_dataset.transform(SSDDefaultTrainTransform(width, height, anchors)),
        batch_size,
        True,
        batchify_fn=batchify_fn,
        last_batch="rollover",
        num_workers=num_workers,
    )
    return train_loader


def ssd_val_dataloader(val_dataset, data_shape=512, batch_size=10, num_workers=0):
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
    """Test on validation dataset."""
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


def save_params(net, best_map, current_map, epoch, save_interval=5, prefix="ssd_512"):
    """
    save parameters of the networks
    """
    current_map = float(current_map)
    if current_map > best_map[0]:
        best_map[0] = current_map
        net.save_parameters("models/{:s}_best.params".format(prefix))
        with open("logs/" + prefix + "_best_map.log", "a") as f:
            f.write("{:04d}:\t{:.4f}\n".format(epoch, current_map))
    if save_interval and epoch % save_interval == 0:
        net.save_parameters(
            "models/{:s}_{:04d}_{:.4f}.params".format(prefix, epoch, current_map)
        )


def get_ctx():
    try:
        a = mx.nd.zeros((1,), ctx=mx.gpu(0))
        ctx = [mx.gpu(0)]
    except:
        ctx = [mx.cpu()]
    return ctx


def val_loss(net, val_data, ctx):
    """Test on validation dataset."""
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
            autograd.backward(sum_loss)
        # since we have already normalized the loss, we don't want to normalize
        # by batch-size anymore
        ce_metric_val.update(0, [l * batch_size for l in cls_loss])
        smoothl1_metric_val.update(0, [l * batch_size for l in box_loss])
        name1, loss1 = ce_metric_val.get()
        name2, loss2 = smoothl1_metric_val.get()
    return loss1, loss2
