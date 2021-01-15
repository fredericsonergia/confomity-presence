import logging
import os 
import argparse
import time
import gluoncv as gcv
import mxnet as mx
import numpy as np
from gluoncv.utils.metrics.voc_detection import VOCMApMetric
from utils.get_results import plot_train
from mxnet import gluon, autograd
from Trainer.trainer_ssd import (load_data_VOC, get_pretrained_model,
                                 train_dataloader, val_dataloader,
                                 validate, save_params, get_ctx, val_loss)
parser = argparse.ArgumentParser(
    description="do training"
)

parser.add_argument(
    "--epoch", default=15, dest="epoch", help="number of training epoch"
)

parser.add_argument(
    "--save-prefix",default='logs/ssd_512', dest="save_prefix", help="number of training epoch"
)

parser.add_argument(
    "--start-epoch", default=0, dest="start_epoch", help="number of training epoch"
)

parser.add_argument(
    "--save-interval", default=5, dest="save_interval", help="interval of epoch for saving models"
)

parser.add_argument(
    "--save-plot", default=True, dest="save_plot", help="saving or not the train curve"
)

parser.add_argument(
    "--suffix-plot", default='1', dest="suffix_plot", help="suffix for the plot file"
)

args = parser.parse_args()


def train(net, ctx, train_data, val_data, val_data2, classes=['cheminee', 'eaf']):
    #set up logging
    
    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_file_path = args.save_prefix + '_train.log'
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(log_file_path)
    logger.addHandler(fh)
    logger.info(args)
    logger.info('Start training from [Epoch {}]'.format(args.start_epoch))
    best_map = [0]
    epochs = np.arange(int(args.epoch))
    ce_loss_list = []
    ce_loss_val = []
    smooth_loss_list = []
    smooth_loss_val = []
    map_list = []
    net.collect_params().reset_ctx(ctx)
    trainer = gluon.Trainer(
        net.collect_params(), 'adam',
        {'learning_rate': 0.001, 'wd': 0.0005})
    mbox_loss = gcv.loss.SSDMultiBoxLoss()
    ce_metric = mx.metric.Loss('CrossEntropy')
    smoothl1_metric = mx.metric.Loss('SmoothL1')
    ce_metric_val = mx.metric.Loss('CrossEntropy')
    smoothl1_metric_val = mx.metric.Loss('SmoothL1')
    for epoch in range(int(args.start_epoch), int(args.epoch)):
        ce_list = np.zeros(shape=(len(train_data)))
        smooth_list = np.zeros(shape=(len(train_data)))
        ce_metric.reset()
        smoothl1_metric.reset()
        tic = time.time()
        btic = time.time()
        net.hybridize(static_alloc=True, static_shape=True)
        for i, batch in enumerate(train_data):
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
                sum_loss, cls_loss, box_loss = mbox_loss(
                    cls_preds, box_preds, cls_targets, box_targets)
                autograd.backward(sum_loss)
            # since we have already normalized the loss, we don't want to normalize
            # by batch-size anymore
            trainer.step(1)
            ce_metric.update(0, [l * batch_size for l in cls_loss])
            smoothl1_metric.update(0, [l * batch_size for l in box_loss])
            name1, loss1 = ce_metric.get()
            name2, loss2 = smoothl1_metric.get()
            logger.info('[Epoch {}][Batch {}], Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}'.format(
                epoch, i, batch_size/(time.time()-btic), name1, loss1, name2, loss2))
            btic = time.time()
            ce_list[i] = loss1
            smooth_list[i] = loss2
        loss1_val, loss2_val = val_loss(net, val_data2, ctx)
        ce_loss_val.append(loss1_val)
        smooth_loss_val.append(loss2_val)
        ce_loss_list.append(np.mean(ce_list))
        logger.info('[Epoch {}] Validation, {}={:.3f}, {}={:.3f}'.format(
            epoch, name1, loss1_val, name2, loss2_val))
        smooth_loss_list.append(np.mean(smooth_list))
        eval_metric = VOCMApMetric(iou_thresh=0.4, class_names=classes)
        map_name, mean_ap = validate(net, val_data,ctx, eval_metric)
        val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
        current_map = mean_ap[1]
        save_params(net, best_map, current_map, epoch, int(args.save_interval))
        map_list.append(current_map)
        logger.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
    return epochs, ce_loss_list, ce_loss_val, smooth_loss_list, smooth_loss_val, map_list


if __name__== '__main__':
    classes = ['cheminee', 'eaf']
    train_dataset, val_dataset = load_data_VOC()
    print(f'taille données train : {len(train_dataset)}', 'taille données val: {len(val_dataset)}')
    net = get_pretrained_model(classes)
    train_data, val_data, val_data2 = train_dataloader(net, train_dataset), val_dataloader(val_dataset), train_dataloader(net, val_dataset)
    ctx = get_ctx()

    epochs, ce_loss_list, ce_loss_val, smooth_loss_list, smooth_loss_val, map_list = train(net, ctx, train_data, val_data, val_data2, classes)
    plot_train(epochs, ce_loss_list, ce_loss_val,
               smooth_loss_list, smooth_loss_val, map_list)